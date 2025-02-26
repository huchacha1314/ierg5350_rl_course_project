import torch
import torch.nn as nn
from multimodal.models.models_utils import (
    duplicate,
    gaussian_parameters,
    rescaleImage,
    product_of_experts,
    sample_gaussian,
    filter_depth,
)
from multimodal.models.base_models.encoders import (
    ProprioEncoder,
    ForceEncoder,
    ImageEncoder,
    DepthEncoder,
)
from multimodal.models.base_models.decoders import (
    OpticalFlowDecoder,
    EeDeltaDecoder,
    ImageDecoder
)

# 实现数据融合
# 将图像、深度图和力传感器数据，并将它们编码成一个低维的多模态特征z
class SensorFusionEncoder(nn.Module):
    def __init__(self, device, z_dim=128, deterministic=True):
        super().__init__()
        #生成的特征维度 128
        self.z_dim = z_dim
        self.device = device
        #deterministic=True 确定性编码
        #deterministic=False 随机性编码
        self.deterministic = deterministic

        self.z_prior_m = torch.nn.Parameter(
            torch.zeros(1, self.z_dim), requires_grad=False
        )
        self.z_prior_v = torch.nn.Parameter(
            torch.ones(1, self.z_dim), requires_grad=False
        )
        self.z_prior = (self.z_prior_m, self.z_prior_v)
        #实例化 ImageEncoder DepthEncoder ForceEncoder
        self.img_encoder = ImageEncoder(self.z_dim)
        self.depth_encoder = DepthEncoder(self.z_dim)
        self.frc_encoder = ForceEncoder(self.z_dim)

        #采取确定性编码
        if deterministic:
            # -----------------------
            # modality fusion network
            # -----------------------
            # 3 Total modalities each (2 * z_dim)
            # 输入维度 [3*2*z_dim,z_dim]
            # 3:3个模态 depth rgb force
            # 2:特征向量 由 均值 方差表示
            self.fusion_fc1 = nn.Sequential(
                nn.Linear(3 * 2 * self.z_dim, self.z_dim), nn.LeakyReLU(0.1, inplace=True)
            )
            self.fusion_fc2 = nn.Sequential(
                nn.Linear(self.z_dim, self.z_dim), nn.LeakyReLU(0.1, inplace=True)
            )
        #对权重和偏置进行初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # kaiming 正态分布的权重初始化，使每一层的方差保持一致，避免梯度爆炸和提督消失
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    #偏执置0
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                #初始权重置1
                m.weight.data.fill_(1)
                #初始权重置0
                m.bias.data.zero_()
    
    def forward(self, vis_in, depth_in, frc_in):
        # assert isinstance(obs_in, tuple), "[SensorFusionEncoder]: Error input, input must be tuple(image, depth and force)"
        # batch size
        if len(vis_in.shape) == 3:
            # 在vis 3维数据前+ 1维 ‘批次’维度
            vis_in = vis_in.unsqueeze(0)
            #扩充至4维
            depth_in = depth_in.unsqueeze(2).unsqueeze(0)
            frc_in = frc_in.unsqueeze(1).unsqueeze(0)

        batch_dim = vis_in.size()[0]

        image = rescaleImage(vis_in)
        depth = filter_depth(depth_in)

        # Get encoded outputs
        img_out, img_out_convs = self.img_encoder(image)
        depth_out, depth_out_convs = self.depth_encoder(depth)
        frc_out = self.frc_encoder(frc_in)

        if self.deterministic:
            '''
            确定多模态：
            
            在确定多模态嵌入中，假设每个模态的编码输出可以直接连接并融合在一起，形成一个统一的表示，而不考虑其具有的概率分布。
            通常通过线性变换、激活函数等操作将每个模态的编码输出连接并处理，最终得到一个确定性的多模态嵌入。

            '''

            # multimodal embedding
            #串联多模态数据
            mm_f1 = torch.cat([img_out, frc_out, depth_out], 1).squeeze()
            # 经过两个线性层 进行数据融合 
            mm_f2 = self.fusion_fc1(mm_f1)
            z = self.fusion_fc2(mm_f2)
            # z是多模态嵌入

            return img_out, img_out_convs, depth_out, depth_out_convs, z
        else:
            '''
            随机多模态：

            在随机多模态嵌入中，假设每个模态（例如图像、深度、力传感器数据等）的编码输出都可以看作是一个概率分布，其具有一定的均值和方差。
            将每个模态的均值和方差连接起来，形成一个整体的多模态分布的参数。
            然后使用这些参数来融合不同模态的信息，通常通过一些融合策略（例如 product of experts）来获得整体的多模态嵌入。
            最后，从融合后的分布中采样，得到一个具体的多模态嵌入。

            '''
            #执行随机多模态嵌入
            # Encoder priors 
            #从经验分布中获取先验均值和方差
            mu_prior, var_prior = self.z_prior

            # Duplicate prior parameters for each data point in the batch
            ###### 解释duplicate 方法#########
            '''
            tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])
            duplicated_tensor = duplicate(tensor, 2)
            将这个张量复制 2 次，即使得每个元素重复 2 次
            tensor([[1, 2],
                    [3, 4],
                    [5, 6],
                    [1, 2],
                    [3, 4],
                    [5, 6]])

            '''
            #将先验扩充到和batch size 一样的大小，并扩展一个新的维度
            mu_prior_resized = duplicate(mu_prior, batch_dim).unsqueeze(2)
            var_prior_resized = duplicate(var_prior, batch_dim).unsqueeze(2)

            # Modality Mean and Variances
            mu_z_img, var_z_img = gaussian_parameters(img_out, dim=1)
            mu_z_frc, var_z_frc = gaussian_parameters(frc_out, dim=1)
            mu_z_depth, var_z_depth = gaussian_parameters(depth_out, dim=1)

            # Tile distribution parameters using concatonation
            m_vect = torch.cat(
                [mu_z_img, mu_z_frc, mu_z_depth, mu_prior_resized], dim=2
            )
            var_vect = torch.cat(
                [var_z_img, var_z_frc, var_z_depth, var_prior_resized],
                dim=2,
            )

            # Fuse modalities mean / variances using product of experts
            #使用product of experts 算法融合不同模态的均值和方差
            #得到整体的多模态均值（mu_z）和方差（var_z）
            mu_z, var_z = product_of_experts(m_vect, var_vect)

            # Sample Gaussian to get latent
            # 生成多模态的分布采样
            z = sample_gaussian(mu_z, var_z, self.device)

            return img_out_convs, img_out, depth_out_convs, depth_out, z

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if "weight" in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if "bias" in name]

# 自监督传感器融合网络，它利用传感器融合编码器的特征表示和动作输入来预测图像重建、深度图重建和接触情况
class MySensorFusionSelfSupervised(nn.Module):
    def __init__(self, device, z_dim=128, action_dim=3, deterministic=True):
        super().__init__()
        self.z_dim = z_dim

        self.deterministic = deterministic

        self.obs_encoder = SensorFusionEncoder(device, self.z_dim, deterministic)

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # fuse observation and activation
        self.o_a_fusion = nn.Sequential(
            nn.Linear(32 + self.z_dim, 128), nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, self.z_dim), nn.LeakyReLU(0.1, inplace=True)
        )

        # color decoder
        self.color_decoder = ImageDecoder(z_dim, mode="color")

        self.depth_decoder = ImageDecoder(z_dim, out_dim=1, mode="depth")

        self.contact_classifier = nn.Sequential(nn.Linear(self.z_dim, 1),
                                                nn.Sigmoid())

    def forward(self, vis_in, depth_in, frc_in, action):
        img_out, img_out_convs, depth_out, depth_out_convs, z = self.obs_encoder(vis_in, depth_in, frc_in)

        act_feat = self.action_encoder(action)

        num_batch = z.shape[0]
        mm_action_feat = self.o_a_fusion(torch.cat([z.view(num_batch, 1, -1), act_feat], 2))

        batch_dim = mm_action_feat.size(0)  # batch size
        tiled_feat = mm_action_feat.view(batch_dim, self.z_dim, 1, 1).expand(-1, -1, 4, 4)

        vis_rec = self.color_decoder(tiled_feat, img_out_convs)
        depth_rec = self.depth_decoder(tiled_feat, depth_out_convs)
        contact_out = self.contact_classifier(mm_action_feat)

        return vis_rec, depth_rec, contact_out

# 传感器融合网络，它接收图像、深度图、力传感器数据、动作数据，并将它们编码成一个低维的多模态特征表示
class SensorFusion(nn.Module):
    """
    #
        Regular SensorFusionNetwork Architecture
        Number of parameters:
        Inputs:
            image:   batch_size x 3 x 128 x 128
            force:   batch_size x 6 x 32
            proprio: batch_size x 8
            action:  batch_size x action_dim
    """

    def __init__(
        self, device, z_dim=128, action_dim=4, encoder=False, deterministic=False
    ):
        super().__init__()

        self.z_dim = z_dim
        self.encoder_bool = encoder
        self.device = device
        self.deterministic = deterministic

        # zero centered, 1 std normal distribution
        self.z_prior_m = torch.nn.Parameter(
            torch.zeros(1, self.z_dim), requires_grad=False
        )
        self.z_prior_v = torch.nn.Parameter(
            torch.ones(1, self.z_dim), requires_grad=False
        )
        self.z_prior = (self.z_prior_m, self.z_prior_v)

        # -----------------------
        # Modality Encoders
        # -----------------------
        self.img_encoder = ImageEncoder(self.z_dim)
        self.depth_encoder = DepthEncoder(self.z_dim)
        self.frc_encoder = ForceEncoder(self.z_dim)
        self.proprio_encoder = ProprioEncoder(self.z_dim)

        # -----------------------
        # Action Encoders
        # -----------------------
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # -----------------------
        # action fusion network
        # -----------------------
        self.st_fusion_fc1 = nn.Sequential(
            nn.Linear(32 + self.z_dim, 128), nn.LeakyReLU(0.1, inplace=True)
        )
        self.st_fusion_fc2 = nn.Sequential(
            nn.Linear(128, self.z_dim), nn.LeakyReLU(0.1, inplace=True)
        )

        if deterministic:
            # -----------------------
            # modality fusion network
            # -----------------------
            # 4 Total modalities each (2 * z_dim)
            self.fusion_fc1 = nn.Sequential(
                nn.Linear(4 * 2 * self.z_dim, 128), nn.LeakyReLU(0.1, inplace=True)
            )
            self.fusion_fc2 = nn.Sequential(
                nn.Linear(self.z_dim, self.z_dim), nn.LeakyReLU(0.1, inplace=True)
            )

        # -----------------------
        # weight initialization
        # -----------------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward_encoder(self, vis_in, frc_in, proprio_in, depth_in, action_in):

        # batch size
        batch_dim = vis_in.size()[0]

        image = rescaleImage(vis_in)
        depth = filter_depth(depth_in)

        # Get encoded outputs
        img_out, img_out_convs = self.img_encoder(image)
        depth_out, depth_out_convs = self.depth_encoder(depth)
        frc_out = self.frc_encoder(frc_in)
        proprio_out = self.proprio_encoder(proprio_in)

        if self.deterministic:
            # multimodal embedding
            mm_f1 = torch.cat([img_out, frc_out, proprio_out, depth_out], 1).squeeze()
            mm_f2 = self.fusion_fc1(mm_f1)
            z = self.fusion_fc2(mm_f2)

        else:
            # Encoder priors
            mu_prior, var_prior = self.z_prior

            # Duplicate prior parameters for each data point in the batch
            mu_prior_resized = duplicate(mu_prior, batch_dim).unsqueeze(2)
            var_prior_resized = duplicate(var_prior, batch_dim).unsqueeze(2)

            # Modality Mean and Variances
            mu_z_img, var_z_img = gaussian_parameters(img_out, dim=1)
            mu_z_frc, var_z_frc = gaussian_parameters(frc_out, dim=1)
            mu_z_proprio, var_z_proprio = gaussian_parameters(proprio_out, dim=1)
            mu_z_depth, var_z_depth = gaussian_parameters(depth_out, dim=1)

            # Tile distribution parameters using concatonation
            m_vect = torch.cat(
                [mu_z_img, mu_z_frc, mu_z_proprio, mu_z_depth, mu_prior_resized], dim=2
            )
            var_vect = torch.cat(
                [var_z_img, var_z_frc, var_z_proprio, var_z_depth, var_prior_resized],
                dim=2,
            )

            # Fuse modalities mean / variances using product of experts
            mu_z, var_z = product_of_experts(m_vect, var_vect)

            # Sample Gaussian to get latent
            z = sample_gaussian(mu_z, var_z, self.device)

        if self.encoder_bool or action_in is None:
            if self.deterministic:
                return img_out, frc_out, proprio_out, depth_out, z
            else:
                return img_out_convs, img_out, frc_out, proprio_out, depth_out, z
        else:
            # action embedding
            act_feat = self.action_encoder(action_in)

            # state-action feature
            mm_act_f1 = torch.cat([z, act_feat], 1)
            mm_act_f2 = self.st_fusion_fc1(mm_act_f1)
            mm_act_feat = self.st_fusion_fc2(mm_act_f2)

            if self.deterministic:
                return img_out_convs, mm_act_feat, z
            else:
                return img_out_convs, mm_act_feat, z, mu_z, var_z, mu_prior, var_prior

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if "weight" in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if "bias" in name]

    def forward(self, *input):
        raise NotImplementedError


class SensorFusionSelfSupervised(SensorFusion):
    """
        Regular SensorFusionNetwork Architecture
        Inputs:
            image:   batch_size x 3 x 128 x 128
            force:   batch_size x 6 x 32
            proprio: batch_size x 8
            action:  batch_size x action_dim
    """

    def __init__(
        self, device, z_dim=128, action_dim=4, encoder=False, deterministic=False
    ):

        super().__init__(device, z_dim, action_dim, encoder, deterministic)

        self.deterministic = deterministic

        # -----------------------
        # optical flow predictor
        # -----------------------
        self.optical_flow_decoder = OpticalFlowDecoder(z_dim)

        # -----------------------
        # ee delta decoder
        # -----------------------
        self.ee_delta_decoder = EeDeltaDecoder(z_dim, action_dim)

        # -----------------------
        # pairing decoder
        # -----------------------
        self.pair_fc = nn.Sequential(nn.Linear(self.z_dim, 1))

        # -----------------------
        # contact decoder
        # -----------------------
        self.contact_fc = nn.Sequential(nn.Linear(self.z_dim, 1))

        # -----------------------
        # weight initialization
        # -----------------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(
        self,
        vis_in,
        frc_in,
        proprio_in,
        depth_in,
        action_in,
    ):

        if self.encoder_bool:
            # returning latent space representation if model is set in encoder mode
            z = self.forward_encoder(vis_in, frc_in, proprio_in, depth_in, action_in)
            return z

        elif action_in is None:
            z = self.forward_encoder(vis_in, frc_in, proprio_in, depth_in, None)
            pair_out = self.pair_fc(z)
            return pair_out

        else:
            if self.deterministic:
                img_out_convs, mm_act_feat, z = self.forward_encoder(
                    vis_in, frc_in, proprio_in, depth_in, action_in
                )
            else:
                img_out_convs, mm_act_feat, z, mu_z, var_z, mu_prior, var_prior = self.forward_encoder(
                    vis_in,
                    frc_in,
                    proprio_in,
                    depth_in,
                    action_in,
                )

        # ---------------- Training Objectives ----------------

        # tile state-action features and append to conv map
        batch_dim = mm_act_feat.size(0)  # batch size
        tiled_feat = mm_act_feat.view(batch_dim, self.z_dim, 1, 1).expand(-1, -1, 2, 2)

        # -------------------------------------#
        # Pairing / Contact / EE Delta Decoder #
        # -------------------------------------#
        pair_out = self.pair_fc(z)
        contact_out = self.contact_fc(mm_act_feat)
        ee_delta_out = self.ee_delta_decoder(mm_act_feat)

        # -------------------------#
        # Optical Flow Prediction #
        # -------------------------#
        optical_flow2, optical_flow2_mask = self.optical_flow_decoder(
            tiled_feat, img_out_convs
        )

        if self.deterministic:
            return (
                pair_out,
                contact_out,
                optical_flow2,
                optical_flow2_mask,
                ee_delta_out,
                z,
            )
        else:
            return (
                pair_out,
                contact_out,
                optical_flow2,
                optical_flow2_mask,
                ee_delta_out,
                z,
                mu_z,
                var_z,
                mu_prior,
                var_prior,
            )


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    encoder = SensorFusionEncoder(device="cuda").cuda()
    encoder.train()

    self_supervise = MySensorFusionSelfSupervised(device="cuda").cuda()
    self_supervise.train()

    for i in range(500):
        vis_in = torch.rand(8, 224, 224, 3).to("cuda")
        depth_in = torch.rand(8, 224, 224, 1).to("cuda")
        frc_in = torch.rand(8, 6, 1).to("cuda")
        act_in = torch.rand(8, 1, 3).to("cuda")

        obs = [vis_in, depth_in, frc_in]

        _, _, _, _, z = encoder(*obs)
        print("z:", z.dtype, z.shape)

        vis_rec, depth_rec, contact_out = self_supervise(vis_in, depth_in, frc_in, act_in)

        print("vis_rec:", vis_rec.dtype, vis_rec.shape)
        print("depth_rec:", depth_rec.dtype, depth_rec.shape)
        print("contact_out", contact_out.dtype, contact_out.shape)


