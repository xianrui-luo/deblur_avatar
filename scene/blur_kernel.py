#from https://github.dev/benhenryL/Deblurring-3D-Gaussian-Splatting

import torch
import torch.nn as nn
import numpy as np


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : i,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

def init_linear_weights(m):
    if isinstance(m, nn.Linear):
        if m.weight.shape[0] in [2, 3]:
            nn.init.xavier_normal_(m.weight, 0.1)
        else:
            nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class GTnet(nn.Module):
    def __init__(self, res_pos=3, res_view=10, num_hidden=3, width=64, pos_delta=False, num_moments=4,feat_cnl=15):
        super().__init__()
        self.pos_delta = pos_delta
        self.num_moments = num_moments

        self.embed_pos, self.embed_pos_cnl = get_embedder(res_pos, 3)
        # self.embed_view, self.embed_view_cnl = get_embedder(res_view, 3)
        self.embed_view, self.embed_view_cnl = get_embedder(res_view, 1)
        in_cnl = self.embed_pos_cnl + self.embed_view_cnl + 7 # 7 for scales and rotations

        # self.feat_cnl = feat_cnl
        # in_cnl+=self.feat_cnl

        hiddens = [nn.Linear(width, width) if i % 2 == 0 else nn.ReLU()
                    for i in range((num_hidden - 1) * 2)]

        self.linears = nn.Sequential(
                nn.Linear(in_cnl, width),
                nn.ReLU(),
                *hiddens,
        ).to("cuda")
        if not pos_delta:   # Defocus
            self.s = nn.Linear(width, 3).to("cuda")
            self.r = nn.Linear(width, 4).to("cuda")
        else:   # Motion
            self.s = nn.Linear(width, 3*(num_moments )).to("cuda")
            self.r = nn.Linear(width, 4*(num_moments )).to("cuda")
            # self.s = nn.Linear(width, 3*(num_moments + 1)).to("cuda")
            # self.r = nn.Linear(width, 4*(num_moments + 1)).to("cuda")
            self.p = nn.Linear(width, 3*num_moments).to("cuda")
            self.mask = nn.Linear(width, 1).to("cuda")


        self.linears.apply(init_linear_weights)
        self.s.apply(init_linear_weights)
        self.r.apply(init_linear_weights)
        if pos_delta:
            self.p.apply(init_linear_weights)

        
            
    # def forward(self, pos, scales, rotations, viewdirs):
    def forward(self, pos, scales, rotations, cam_idxs):
        pos_delta = None
        pos = self.embed_pos(pos)
        # viewdirs = self.embed_view(viewdirs)
        viewdirs = self.embed_view(cam_idxs)



        # if feats==None:
        #     x = torch.cat([pos, viewdirs, scales, rotations,torch.zeros(pos.shape[0],self.feat_cnl)], dim=-1)
        # else:
        #     x= torch.cat([pos, viewdirs, scales, rotations,feats.view(pos.shape[0],-1)], dim=-1)
        x = torch.cat([pos, viewdirs, scales, rotations], dim=-1)
        x1 = self.linears(x)

        scales_delta = self.s(x1)
        rotations_delta = self.r(x1)

        if self.pos_delta:
            pos_delta = self.p(x1)
        mask=self.mask(x1)
        mask=torch.sigmoid(mask)

        return scales_delta, rotations_delta, pos_delta,mask
        

# ... existing code ...

class smpl_net(nn.Module):
    def __init__(self, input_dim=72, hidden_dim=256, num_layers=3):
        """
        Initialize the smpl_net for predicting SMPL relative rotations.
        
        :param input_dim: Dimension of input SMPL parameters
        :param seq_length: Number of frames in the output sequence
        :param hidden_dim: Dimension of hidden layers
        :param num_layers: Number of hidden layers
        """
        super().__init__()
        
        # self.seq_length = seq_length
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Output layer for relative rotations (assuming 24 joints with 3 rotation parameters each)
        # self.output_layer = nn.Linear(hidden_dim, seq_length * 24 * 3)
        self.output_layer = nn.Linear(hidden_dim, 24 * 3)
        
        # Activation function
        self.activation = nn.ReLU()
        
    def forward(self, smpl_params):
        """
        Forward pass of the network.
        
        :param smpl_params: Input SMPL parameters (batch_size, input_dim)
        :return: Sequence of relative rotations (batch_size, seq_length, 24, 3)
        """
        x = self.activation(self.input_layer(smpl_params))
        
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        
        # Output relative rotations
        relative_rotations = self.output_layer(x)
        
        # Reshape the output to (batch_size, seq_length, 24, 3)
        # return relative_rotations.view(-1, self.seq_length, 24, 3)
        return relative_rotations.view(-1, 24, 3)


class BlurKernel(nn.Module):
    def __init__(self, num_img, img_embed=32, ks=17, not_use_rgbd=False,not_use_pe=False):
        super().__init__()

        ## use image or smpl as feature

        self.num_img = num_img

        self.img_embed_cnl = img_embed

        self.min_freq, self.max_freq, self.num_frequencies = 0.0, 3.0, 4

        self.embedding_camera = nn.Embedding(self.num_img, self.img_embed_cnl)

        print('this is single res kernel', ks)
        
        self.not_use_rgbd = not_use_rgbd
        self.not_use_pe = not_use_pe
        print('single res: not_use_rgbd', self.not_use_rgbd, 'not_use_pe', self.not_use_pe)
        rgd_dim = 0 if self.not_use_rgbd else 32
        pe_dim = 0 if self.not_use_pe else 16

        self.mlp_base_mlp = torch.nn.Sequential(
            torch.nn.Conv2d(32+pe_dim+rgd_dim, 64, 1, bias=False), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 1, bias=False), torch.nn.ReLU(),
            )
        
        self.mlp_head1 = torch.nn.Conv2d(64, ks**2, 1, bias=False)
        self.mlp_mask1 = torch.nn.Conv2d(64, 1, 1, bias=False)

        self.conv_rgbd = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 5,padding=2), torch.nn.ReLU(), torch.nn.InstanceNorm2d(64),
            torch.nn.Conv2d(64, 64, 5,padding=2), torch.nn.ReLU(), torch.nn.InstanceNorm2d(64),
            torch.nn.Conv2d(64, 32, 3,padding=1)
            )

    def forward(self, img_idx, pos_enc, img):
        img_embed = self.embedding_camera(img_idx)[None, None]
        img_embed = img_embed.expand(pos_enc.shape[0],pos_enc.shape[1],pos_enc.shape[2],img_embed.shape[-1])

        if self.not_use_pe:
            inp = img_embed.permute(0,3,1,2)
        else:
            inp = torch.cat([img_embed,pos_enc],-1).permute(0,3,1,2)

        if self.not_use_rgbd:
            feat = self.mlp_base_mlp(inp)
        else:
            rgbd_feat = self.conv_rgbd(img)
            feat = self.mlp_base_mlp(torch.cat([inp,rgbd_feat],1))

        weight = self.mlp_head1(feat)
        mask = self.mlp_mask1(feat)

        weight = torch.softmax(weight, dim=1)
        mask = torch.sigmoid(mask)

        return weight, mask



class BlurKernel_mask(nn.Module):
    def __init__(self, num_img, img_embed=32, ks=17, not_use_rgbd=False,not_use_pe=False):
        super().__init__()

        ## use image or smpl as feature

        self.num_img = num_img

        self.img_embed_cnl = img_embed

        self.min_freq, self.max_freq, self.num_frequencies = 0.0, 3.0, 4

        self.embedding_camera = nn.Embedding(self.num_img, self.img_embed_cnl)

        # print('this is single res kernel', ks)
        
        self.not_use_rgbd = not_use_rgbd
        self.not_use_pe = not_use_pe
        # print('single res: not_use_rgbd', self.not_use_rgbd, 'not_use_pe', self.not_use_pe)
        rgd_dim = 0 if self.not_use_rgbd else 32
        pe_dim = 0 if self.not_use_pe else 16

        self.mlp_base_mlp = torch.nn.Sequential(
            torch.nn.Conv2d(32+pe_dim+rgd_dim, 64, 1, bias=False), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 1, bias=False), torch.nn.ReLU(),
            )
        self.mlp_mask1 = torch.nn.Conv2d(64, 1, 1, bias=False)

        self.conv_rgbd = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 5,padding=2), torch.nn.ReLU(), torch.nn.InstanceNorm2d(64),
            torch.nn.Conv2d(64, 64, 5,padding=2), torch.nn.ReLU(), torch.nn.InstanceNorm2d(64),
            torch.nn.Conv2d(64, 32, 3,padding=1)
            )
            

    # def forward(self, img_idx, pos_enc,pose_diff, img):
    def forward(self, img_idx, pos_enc, img):
        # img_embed = self.embedding_camera(torch.LongTensor([img_idx]).cuda())[None, None]
        img_embed = self.embedding_camera(img_idx)[None, None]
        img_embed = img_embed.expand(pos_enc.shape[0],pos_enc.shape[1],pos_enc.shape[2],img_embed.shape[-1])
        inp=torch.cat([img_embed,pos_enc],-1).permute(0,3,1,2)

        if self.not_use_rgbd:
            feat = self.mlp_base_mlp(inp)
        else:
            rgbd_feat = self.conv_rgbd(img)
            feat = self.mlp_base_mlp(torch.cat([inp,rgbd_feat],1))

        # weight = self.mlp_head1(feat)
        mask = self.mlp_mask1(feat)
        mask = torch.sigmoid(mask)
        return mask
class mixture(nn.Module):
    def __init__(self, num_img, img_embed=32, not_use_rgbd=False,not_use_pe=False):
        super().__init__()

        ## use image or smpl as feature

        self.num_img = num_img

        self.img_embed_cnl = img_embed

        self.min_freq, self.max_freq, self.num_frequencies = 0.0, 3.0, 4

        self.embedding_camera = nn.Embedding(self.num_img, self.img_embed_cnl)
        
        self.not_use_rgbd = not_use_rgbd
        self.not_use_pe = not_use_pe
        print('single res: not_use_rgbd', self.not_use_rgbd, 'not_use_pe', self.not_use_pe)
        rgd_dim = 0 if self.not_use_rgbd else 32
        pe_dim = 0 if self.not_use_pe else 16

        self.mlp_base_mlp = torch.nn.Sequential(
            torch.nn.Conv2d(32+pe_dim+rgd_dim, 64, 1, bias=False), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 1, bias=False), torch.nn.ReLU(),
            )
        
        # self.mlp_head1 = torch.nn.Conv2d(64, ks**2, 1, bias=False)
        self.mlp_mask1 = torch.nn.Conv2d(64, 4, 1, bias=False)

        self.conv_rgbd = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 5,padding=2), torch.nn.ReLU(), torch.nn.InstanceNorm2d(64),
            torch.nn.Conv2d(64, 64, 5,padding=2), torch.nn.ReLU(), torch.nn.InstanceNorm2d(64),
            torch.nn.Conv2d(64, 32, 3,padding=1)
            )

    def forward(self, img_idx, pos_enc, img):
        # img_embed = self.embedding_camera(torch.LongTensor([img_idx]).cuda())[None, None]
        img_embed = self.embedding_camera(img_idx)[None, None]
        img_embed = img_embed.expand(pos_enc.shape[0],pos_enc.shape[1],pos_enc.shape[2],img_embed.shape[-1])

        if self.not_use_pe:
            inp = img_embed.permute(0,3,1,2)
        else:
            inp = torch.cat([img_embed,pos_enc],-1).permute(0,3,1,2)

        if self.not_use_rgbd:
            feat = self.mlp_base_mlp(inp)
        else:
            rgbd_feat = self.conv_rgbd(img)
            feat = self.mlp_base_mlp(torch.cat([inp,rgbd_feat],1))
        info = self.mlp_mask1(feat)
        info[:,:2]=torch.softmax(info[:,:2], dim=1)


        return info



