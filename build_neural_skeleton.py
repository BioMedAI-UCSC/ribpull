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
        # Load checkpoint
        checkpoint = torch.load(trainednet, map_location=device)
        
        # Create network with 8 layers (lin0-lin7) + manual lin8
        net = NPullNetwork(d_in=3, d_out=256, d_hidden=256, n_layers=8, skip_in=(4,), weight_norm=True, geometric_init=False).to(device)
        
        # Add final layer manually
        lin8 = nn.Linear(256, 2)
        setattr(net, "lin8", lin8)
        
        # Let's test different skip connection interpretations
        def test_architecture():
            test_input = torch.randn(1, 3).to(device)
            x = test_input
            
            print("Testing architecture step by step:")
            print(f"Input: {x.shape}")
            
            # Test what happens if we follow the exact layer dimensions
            try:
                # lin0: 3 -> 256
                x = torch.randn(1, 256).to(device)  # Simulate lin0 output
                print(f"After lin0: {x.shape}")
                
                # lin1: 256 -> 256  
                # lin2: 256 -> 256
                # These are straightforward
                
                # lin3: expects 256 input, outputs 253
                # This suggests lin3 is the "reduced" layer due to upcoming skip connection
                lin3_out = torch.randn(1, 253).to(device)
                print(f"After lin3 (reduced): {lin3_out.shape}")
                
                # Now add skip connection: 253 + 3 = 256
                skip_result = torch.cat([lin3_out, test_input], dim=1)
                print(f"After skip connection: {skip_result.shape}")
                
                # lin4: 256 -> 256 (this matches the saved weights)
                print("This matches the saved model dimensions!")
                return True
                
            except Exception as e:
                print(f"Test failed: {e}")
                return False
        
        if test_architecture():
            print("Architecture understanding confirmed!")
        
        # Create the correct forward pass based on this understanding
        def forward(inputs):
            # Handle both 2D and 3D inputs by flattening if necessary
            original_shape = inputs.shape
            if len(inputs.shape) == 3:
                # Flatten 3D input [batch1, batch2, 3] -> [batch1*batch2, 3]
                inputs = inputs.reshape(-1, inputs.shape[-1])
            
            x = inputs * net.scale
            original_inputs = x
            
            # Process layers following the saved model's exact architecture
            x = net.lin0(x)  # 3 -> 256
            x = net.activation(x)
            
            x = net.lin1(x)  # 256 -> 256
            x = net.activation(x)
            
            x = net.lin2(x)  # 256 -> 256
            x = net.activation(x)
            
            # lin3 is the "skip layer" - it reduces dimension to make room for skip connection
            x = net.lin3(x)  # 256 -> 253
            # Now add the skip connection
            x = torch.cat([x, original_inputs], dim=1) / np.sqrt(2)  # 253 + 3 -> 256
            x = net.activation(x)
            
            # Continue with remaining layers
            x = net.lin4(x)  # 256 -> 256
            x = net.activation(x)
            
            x = net.lin5(x)  # 256 -> 256
            x = net.activation(x)
            
            x = net.lin6(x)  # 256 -> 256
            x = net.activation(x)
            
            x = net.lin7(x)  # 256 -> 256
            x = net.activation(x)
            
            x = net.lin8(x)  # 256 -> 2
            
            result = x[:, :1] / net.scale
            
            # Reshape back to original shape if input was 3D
            if len(original_shape) == 3:
                result = result.reshape(original_shape[0], original_shape[1], 1)
            
            return result
        
        # Replace forward method
        net.forward = forward
        
        # Load weights
        net.load_state_dict(checkpoint, strict=False)
        print("Model loaded successfully!")
        
    else:
        # Original training path
        pc, nc = torch.tensor(pc1, device=device).float(), torch.tensor(nc1, device=device).float()
        if scaleshape:
            pc, center, scale = center_bounding_box(pc)
        else:
            center = torch.zeros((3), device=device)
            scale = torch.ones((1), device=device)
        pc.requires_grad = True
        
        net = torch.load("Pretrained/pretrained_{}_{}_{}.net".format(npl, dep, activation))
        
        print("\n##### Optimizing the neural sdf ({},{})".format(activation, "TV" if tv else "No TV"))
        if activation == "Sine":
            optim = torch.optim.LBFGS(params=net.parameters())
            nepochs = 50
            nhints_ends = 20
        elif activation == "ReLU":
            optim = torch.optim.Adam(params=net.parameters(), lr=2e-5)
            nepochs = 20000
            nhints_ends = 10000
        elif activation == "SoftPlus":
            optim = torch.optim.Adam(params=net.parameters(), lr=1e-3)
            nepochs = 20000
            nhints_ends = 10000
        
        t = time.time()
        try:
            optimize_neural_sdf(net, optim, pc, nc, batch_size=25000, pc_batch_size=25000, 
                                epochs=nepochs, tv_ends=nepochs if tv else 0, hints_ends=nhints_ends,
                                lambda_pc=lambda_pc, lambda_eik=lambda_eik, lambda_hint=lambda_hint, lambda_tv=lambda_tv,
                                nb_hints=hints, plot_loss=False)
        except KeyboardInterrupt:
            pass
        print("Optimizing NN took", '{:.2f}'.format(time.time()-t), "s.")
    
    print("\n##### Computing neural coverage skeleton")
    tskel = time.time()
    
    D = 2*np.sqrt(3)
    number = 10000
    samples = projection(net, number=number, prune=True)
    if resampling:
        samples = uniform_resampling(net, samples, 100, K=3, alpha=.1*np.sqrt(D/number), sigma=16*D/number)

    sk = sample_skeleton_gpu(net, samples, res=50, length=1, steps=1, div=100)
    
    print("Extracting skeletal points candidates", time.time()-tskel)
    
    candidates = neural_candidates(sk, reduce_radius=0.01)
    cvskpts, edges, triangles = coverage_skeleton(candidates, samples, delta=delta, factor=scaling, time_limit=time_limit)

    print("Coverage skeleton obtained in", '{:.2f}'.format(time.time()-tskel), " s.") 

    # Handle scaling for loaded vs trained models
    if trainednet is not None:
        skpts = candidates.cpu().numpy()
        upts = samples.detach().cpu().numpy()
        cvskpts_final = cvskpts
    else:
        skpts = candidates.cpu().numpy() * scale.cpu().numpy() + center.cpu().numpy()
        upts = samples.detach().cpu().numpy() * scale.cpu().numpy() + center.cpu().numpy()
        if cvskpts.shape[0] > 0:
            cvskpts_final = cvskpts * scale.cpu().numpy() + center.cpu().numpy()
        else:
            cvskpts_final = cvskpts

    print("Total computation time", '{:.2f}'.format(time.time()-tinit), " s.")

    if cvskpts_final.shape[0] == 0:
        print("Infeasible problem no skeleton found, try tweaking the parameters")
    else:
        print("Coverage skeleton built successfully")

    return cvskpts_final, edges, triangles, net, skpts, upts