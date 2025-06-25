import torch
import torch.nn as nn
import numpy as np
import time
from nn import optimize_neural_sdf
from skeleton import sample_skeleton_gpu, coverage_skeleton, neural_candidates
from geometry import projection, uniform_resampling, center_bounding_box, get_bounding_box

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NPullNetwork(nn.Module):
    def __init__(self, d_in, d_out, d_hidden, n_layers, skip_in=(4,), multires=0, bias=0.5, scale=1, 
                 geometric_init=True, weight_norm=True, inside_outside=False):
        super(NPullNetwork, self).__init__()
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.embed_fn_fine = None
        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale
        
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            
            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
            
        self.activation = nn.ReLU()
        
    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)
        return x / self.scale
    
    def sdf(self, x):
        return self.forward(x)

def build_neural_skeleton(pc1, nc1, tv=True, activation="Sine", npl=64, dep=6, hints=1000, 
                          delta=0.06, scaling=None, lambda_pc=100, lambda_eik=2e2, lambda_hint=1e2, 
                          lambda_tv=2e1, resampling=True, trainednet=None, time_limit=120, scaleshape=True):

    print(delta)
    tinit = time.time()
    
    if trainednet is not None:
        # Load pre-trained model
        checkpoint = torch.load(trainednet, map_location=device)
        
        # Create network architecture matching the saved model
        net = NPullNetwork(d_in=3, d_out=256, d_hidden=256, n_layers=8, skip_in=(4,), 
                          weight_norm=True, geometric_init=False).to(device)
        
        # Add final output layer
        lin8 = nn.Linear(256, 2)
        setattr(net, "lin8", lin8)
        
        # Custom forward pass for pre-trained model with skip connections
        def forward(inputs):
            # Handle both 2D and 3D inputs
            original_shape = inputs.shape
            if len(inputs.shape) == 3:
                inputs = inputs.reshape(-1, inputs.shape[-1])
            
            x = inputs * net.scale
            original_inputs = x
            
            # Standard layers
            x = net.lin0(x)  # 3 -> 256
            x = net.activation(x)
            x = net.lin1(x)  # 256 -> 256
            x = net.activation(x)
            x = net.lin2(x)  # 256 -> 256
            x = net.activation(x)
            
            # Skip connection layer: reduce then concatenate
            x = net.lin3(x)  # 256 -> 253
            x = torch.cat([x, original_inputs], dim=1) / np.sqrt(2)  # 253 + 3 -> 256
            x = net.activation(x)
            
            # Remaining layers
            x = net.lin4(x)
            x = net.activation(x)
            x = net.lin5(x)
            x = net.activation(x)
            x = net.lin6(x)
            x = net.activation(x)
            x = net.lin7(x)
            x = net.activation(x)
            x = net.lin8(x)  # 256 -> 2
            
            result = x[:, :1] / net.scale  # Return SDF component only
            
            # Reshape back if input was 3D
            if len(original_shape) == 3:
                result = result.reshape(original_shape[0], original_shape[1], 1)
            
            return result
        
        net.forward = forward
        net.load_state_dict(checkpoint, strict=False)
        print("Pre-trained model loaded successfully!")
        
    else:
        # Train new model from scratch
        pc, nc = torch.tensor(pc1, device=device).float(), torch.tensor(nc1, device=device).float()
        
        if scaleshape:
            pc, center, scale = center_bounding_box(pc)
        else:
            center = torch.zeros((3), device=device)
            scale = torch.ones((1), device=device)
        pc.requires_grad = True
        
        net = torch.load("Pretrained/pretrained_{}_{}_{}.net".format(npl, dep, activation))
        
        print("\n##### Optimizing neural SDF ({},{})".format(activation, "TV" if tv else "No TV"))
        
        # Set optimizer and training parameters based on activation
        if activation == "Sine":
            optim = torch.optim.LBFGS(params=net.parameters())
            nepochs, nhints_ends = 50, 20
        elif activation == "ReLU":
            optim = torch.optim.Adam(params=net.parameters(), lr=2e-5)
            nepochs, nhints_ends = 20000, 10000
        elif activation == "SoftPlus":
            optim = torch.optim.Adam(params=net.parameters(), lr=1e-3)
            nepochs, nhints_ends = 20000, 10000
        
        t = time.time()
        try:
            optimize_neural_sdf(net, optim, pc, nc, batch_size=25000, pc_batch_size=25000, 
                                epochs=nepochs, tv_ends=nepochs if tv else 0, hints_ends=nhints_ends,
                                lambda_pc=lambda_pc, lambda_eik=lambda_eik, lambda_hint=lambda_hint, 
                                lambda_tv=lambda_tv, nb_hints=hints, plot_loss=False)
        except KeyboardInterrupt:
            pass
        print("Neural SDF optimization completed in {:.2f}s".format(time.time()-t))
    
    # Extract skeleton from neural SDF
    print("\n##### Computing neural coverage skeleton")
    tskel = time.time()
    
    D = 2*np.sqrt(3)
    number = 10000
    samples = projection(net, number=number, prune=True)
    if resampling:
        samples = uniform_resampling(net, samples, 100, K=3, alpha=.1*np.sqrt(D/number), sigma=16*D/number)

    sk = sample_skeleton_gpu(net, samples, res=50, length=1, steps=1, div=100)
    candidates = neural_candidates(sk, reduce_radius=0.01)
    cvskpts, edges, triangles = coverage_skeleton(candidates, samples, delta=delta, factor=scaling, time_limit=time_limit)

    print("Coverage skeleton obtained in {:.2f}s".format(time.time()-tskel))

    # Handle coordinate scaling
    if trainednet is not None:
        # Pre-trained model: no scaling needed
        skpts = candidates.cpu().numpy()
        upts = samples.detach().cpu().numpy()
        cvskpts_final = cvskpts
    else:
        # Trained model: apply inverse scaling
        skpts = candidates.cpu().numpy() * scale.cpu().numpy() + center.cpu().numpy()
        upts = samples.detach().cpu().numpy() * scale.cpu().numpy() + center.cpu().numpy()
        if cvskpts.shape[0] > 0:
            cvskpts_final = cvskpts * scale.cpu().numpy() + center.cpu().numpy()
        else:
            cvskpts_final = cvskpts

    print("Total computation time: {:.2f}s".format(time.time()-tinit))

    if cvskpts_final.shape[0] == 0:
        print("Warning: No skeleton found. Try adjusting parameters.")
    else:
        print("Skeleton extraction completed successfully!")

    return cvskpts_final, edges, triangles, net, skpts, upts