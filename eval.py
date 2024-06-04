import torch
import numpy as np
import pyrealsense2 as rs
import cv2
from dynamixel_sdk import *
from network import ConditionalUnet1D, replace_bn_with_gn, get_resnet
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# Define constants and configuration
ADDR_MX_TORQUE_ENABLE = 24
ADDR_MX_GOAL_POSITION = 30
ADDR_MX_PRESENT_POSITION = 36
LEN_MX_GOAL_POSITION = 4
PROTOCOL_VERSION = 1.0
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
MODEL_PATH = 'path/to/your/checkpoint.pt'

# Configuration
config = {
    'pred_horizon': 16,
    'obs_horizon': 2,
    'action_horizon': 8,
    'num_diffusion_iters': 100,
    'batch_size': 64,
    'num_workers': 4,
    'num_epochs': 100,
    'action_dim': 4
}

# Motor control class
class Dynamixel:
    def __init__(self, port, baudrate, dynamixel_ids):
        self.port = port
        self.baudrate = baudrate
        self.dynamixel_ids = dynamixel_ids
        self.portHandler = PortHandler(self.port)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)
        if not self.portHandler.openPort():
            raise Exception("Failed to open the port")
        if not self.portHandler.setBaudRate(self.baudrate):
            raise Exception("Failed to set baud rate")
        for dxl_id in self.dynamixel_ids:
            self._enable_torque(dxl_id)
        self.groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, ADDR_MX_GOAL_POSITION, LEN_MX_GOAL_POSITION)

    def _enable_torque(self, dxl_id):
        self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)

    def get_motor_data(self):
        motor_positions = np.zeros(len(self.dynamixel_ids))
        for i, dxl_id in enumerate(self.dynamixel_ids):
            dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler, dxl_id, ADDR_MX_PRESENT_POSITION)
            if dxl_comm_result != COMM_SUCCESS:
                print(f"Failed to read present position for Dynamixel ID {dxl_id}")
            elif dxl_error != 0:
                print(f"Dynamixel error {dxl_error} for ID {dxl_id}")
            motor_positions[i] = (dxl_present_position / 4095) * 360
        return motor_positions

    def send_angles_to_dynamixels(self, angles):
        for i, dxl_id in enumerate(self.dynamixel_ids):
            position_value = int(angles[i] / 360 * 4095)
            param_goal_position = [DXL_LOBYTE(DXL_LOWORD(position_value)),
                                   DXL_HIBYTE(DXL_LOWORD(position_value)),
                                   DXL_LOBYTE(DXL_HIWORD(position_value)),
                                   DXL_HIBYTE(DXL_HIWORD(position_value))]
            if not self.groupSyncWrite.addParam(dxl_id, param_goal_position):
                raise Exception(f"Failed to add param for Dynamixel ID {dxl_id}")
        if self.groupSyncWrite.txPacket() != COMM_SUCCESS:
            raise Exception("Failed to write goal positions")
        self.groupSyncWrite.clearParam()

    def close(self):
        for dxl_id in self.dynamixel_ids:
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)
        self.portHandler.closePort()

# Camera class
class RealsenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(self.config)

    def get_image(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        image = np.asanyarray(color_frame.get_data())
        return image

    def stop(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()

# Model loading function
def load_model(model_path, device):
    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)
    vision_feature_dim = 512
    lowdim_obs_dim = 4
    obs_dim = vision_feature_dim + lowdim_obs_dim
    noise_pred_net = ConditionalUnet1D(
        input_dim=config['action_dim'],
        global_cond_dim=obs_dim * config['obs_horizon']
    )
    nets = {
        'vision_encoder': vision_encoder.to(device),
        'noise_pred_net': noise_pred_net.to(device)
    }
    checkpoint = torch.load(model_path, map_location=device)
    nets['vision_encoder'].load_state_dict(checkpoint['model_state_dict']['vision_encoder'])
    nets['noise_pred_net'].load_state_dict(checkpoint['model_state_dict']['noise_pred_net'])
    normalization_stats = checkpoint['normalization_stats']
    return nets, normalization_stats

# Create noise scheduler function
def create_noise_scheduler(num_diffusion_iters):
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    return noise_scheduler

# Model inference function
def run_inference(model, realsense_image, motor_positions, device, noise_scheduler, normalization_stats):
    model['vision_encoder'].eval()
    model['noise_pred_net'].eval()
    with torch.no_grad():
        # Preprocess inputs
        realsense_image = torch.tensor(realsense_image, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
        motor_positions = torch.tensor(motor_positions, dtype=torch.float32, device=device).unsqueeze(0)

        # Normalize inputs
        realsense_image /= 255.0
        motor_positions = (motor_positions - normalization_stats['agent_pos']['min']) / (normalization_stats['agent_pos']['max'] - normalization_stats['agent_pos']['min'])

        # Get image features
        image_features = model['vision_encoder'](realsense_image)
        obs_features = torch.cat([image_features, motor_positions], dim=-1)
        obs_cond = obs_features.flatten(start_dim=1)

        # Initialize action from Gaussian noise
        noisy_action = torch.randn(motor_positions.shape, device=device)
        naction = noisy_action

        # Initialize scheduler
        noise_scheduler.set_timesteps(config['num_diffusion_iters'])

        for k in noise_scheduler.timesteps:
            # Predict noise
            noise_pred = model['noise_pred_net'](sample=naction, timestep=k, global_cond=obs_cond)

            # Inverse diffusion step (remove noise)
            naction = noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

        # Unnormalize actions
        naction = naction.squeeze(0).cpu().numpy()
        predicted_actions = (naction * (normalization_stats['action']['max'] - normalization_stats['action']['min']) + normalization_stats['action']['min'])

        return predicted_actions

# Main function
def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dynamixel_ids = [0, 1, 2, 3]  # Example Dynamixel IDs
    dynamixel_port = 'COM4'
    dynamixel_baudrate = 2000000

    # Initialize camera and motors
    realsense = RealsenseCamera()
    dynamixel = Dynamixel(dynamixel_port, dynamixel_baudrate, dynamixel_ids)

    # Load model
    model, normalization_stats = load_model(MODEL_PATH, device)
    noise_scheduler = create_noise_scheduler(config['num_diffusion_iters'])

    try:
        while True:
            # Capture observation from Realsense camera
            realsense_image = realsense.get_image()
            if realsense_image is None:
                continue

            # Capture motor positions
            motor_positions = dynamixel.get_motor_data()

            # Run inference
            predicted_angles = run_inference(model, realsense_image, motor_positions, device, noise_scheduler, normalization_stats)

            # Send commands to motors
            dynamixel.send_angles_to_dynamixels(predicted_angles)

            # Display camera feed
            cv2.imshow('Realsense Camera', realsense_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass

    finally:
        realsense.stop()
        dynamixel.close()

if __name__ == '__main__':
    main()
