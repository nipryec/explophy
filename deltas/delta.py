##################################
#
# Little script to simulate delta formation
#
# version : 22/12/2024
#
# Author : nipryec
#

###
# Imports

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

###
# User settings

# size of the environment
dim = 50  # Increased for better visualization
height = 30  # Height of the environment

# duration of the experiment
duration = 100
# saving points
save_time = [np.exp(i) for i in range(int(np.log(duration)))]

# Simulation parameters
sediment_input = 0.5  # Amount of sediment input per step
flow_threshold = 0.05  # Minimum height difference for flow
erosion_rate = 0.1  # Rate of erosion
slope = 0.1  # Slope of the terrain
flow_speed_threshold = 0.2  # Threshold for slow flow areas
lateral_deposit_rate = 0.3  # Rate of deposition in lateral slow-flow areas
initial_height = 2.0  # Initial height of the terrain
bank_slope_width = 5  # Width parameter for the bank slopes

###
# dev settings

# compute or not, not available yet
# compute = True

###
# initialization

# Initialize ground with height values and create a slope
ground = np.ones((height, dim)) * initial_height
for i in range(height):
	ground[i, :] = initial_height + (height - i) * slope  # Base slope from top to bottom

# Create initial river channel with smooth banks
mid = dim//2
channel_width = 3
x = np.arange(dim)
for i in range(height):
	# Create Gaussian-like banks
	bank_profile = initial_height + (height - i) * slope - 1.5 * np.exp(-(x - mid)**2 / (2 * bank_slope_width**2))
	ground[i, :] = bank_profile
	# Ensure flat river bed in the center
	ground[i, mid-channel_width//2:mid+channel_width//2+1] = (height - i) * slope

# Initialize sediment concentration and flow velocity
sediment = np.zeros((height, dim))
flow_velocity = np.zeros((height, dim))
sediment[0, mid] = sediment_input  # Add sediment at the top of channel

def update_delta(ground, sediment):
	"""Update the delta formation for one time step."""
	new_ground = ground.copy()
	new_sediment = sediment.copy()
	flow_velocity = np.zeros((height, dim))  # Reset flow velocity field
	
	# Calculate flow velocity field (simplified)
	for i in range(height-1):
		for j in range(1, dim-1):
			dh = new_ground[i, j] - new_ground[i+1, j]
			flow_velocity[i, j] = max(0, dh / slope)  # Simplified velocity based on slope
			
			# Higher velocity in the channel
			if abs(j - mid) <= channel_width//2:
				flow_velocity[i, j] *= 1.5
	
	# Process each cell from top to bottom
	for i in range(height-1):
		for j in range(1, dim-1):
			if new_sediment[i, j] < 0.001:
				continue
			
			# Calculate height differences
			dh_down = new_ground[i, j] - new_ground[i+1, j]
			dh_left = new_ground[i, j] - new_ground[i, j-1]
			dh_right = new_ground[i, j] - new_ground[i, j+1]
			
			# Main flow transport
			flow_occurred = False
			if dh_down > flow_threshold:
				# Transport sediment downstream
				transport_rate = min(0.9, flow_velocity[i, j])
				new_sediment[i+1, j] += new_sediment[i, j] * transport_rate
				new_sediment[i, j] *= (1 - transport_rate)
				flow_occurred = True
			
			# Lateral spreading and deposition
			if new_sediment[i, j] > 0.001:
				# Check for slow flow areas on sides
				for dx in [-1, 1]:
					neighbor_j = j + dx
					if 0 <= neighbor_j < dim:
						if flow_velocity[i, neighbor_j] < flow_speed_threshold:
							# Deposit more in slow flow areas
							lateral_deposit = new_sediment[i, j] * lateral_deposit_rate
							new_ground[i, neighbor_j] += lateral_deposit
							new_sediment[i, j] -= lateral_deposit
				
				# Reduced deposition in main channel
				if abs(j - mid) <= channel_width//2:
					deposit_rate = erosion_rate * 0.2  # Reduced deposition in channel
				else:
					deposit_rate = erosion_rate  # Normal deposition outside channel
				
				deposit = new_sediment[i, j] * deposit_rate
				new_ground[i, j] += deposit
				new_sediment[i, j] -= deposit
	
	# Add new sediment at source
	new_sediment[0, mid] = sediment_input
	
	return new_ground, new_sediment

# Create figure for animation
fig = plt.figure(figsize=(15, 5))

# Create three square subplots
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)

plt.suptitle('Delta Formation Simulation', fontsize=14)

# Initialize plots with square aspect ratio
# Force the display to be square by setting both dimensions equal
extent = [-0.5, dim-0.5, height-0.5, -0.5]  # This makes the plot square
im1 = ax1.imshow(ground, cmap='terrain', extent=extent)
im2 = ax2.imshow(sediment, cmap='YlOrRd', extent=extent)
im3 = ax3.imshow(flow_velocity, cmap='Blues', extent=extent)

# Add colorbars
plt.colorbar(im1, ax=ax1, label='Ground Height')
plt.colorbar(im2, ax=ax2, label='Sediment Concentration')
plt.colorbar(im3, ax=ax3, label='Flow Velocity')

# Set titles
ax1.set_title('Ground Height')
ax2.set_title('Sediment Distribution')
ax3.set_title('Flow Velocity')

# Remove axis ticks for cleaner visualization
for ax in [ax1, ax2, ax3]:
    ax.set_xticks([])
    ax.set_yticks([])
    # Force aspect ratio to be equal
    ax.set_aspect('equal', adjustable='box')

# Adjust spacing between plots
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Animation update function
def animate(frame):
	global ground, sediment
	
	# Update simulation state
	ground, sediment = update_delta(ground, sediment)
	
	# Calculate flow velocity for visualization
	flow_vel = np.zeros((height, dim))
	for i in range(height-1):
		for j in range(1, dim-1):
			dh = ground[i, j] - ground[i+1, j]
			flow_vel[i, j] = max(0, dh / slope)
			if abs(j - mid) <= channel_width//2:
				flow_vel[i, j] *= 1.5
	
	# Update plots
	im1.set_array(ground)
	im2.set_array(sediment)
	im3.set_array(flow_vel)
	
	# Update titles
	ax1.set_title(f'Ground Height (Step {frame+1})')
	ax2.set_title(f'Sediment Distribution (Step {frame+1})')
	ax3.set_title(f'Flow Velocity (Step {frame+1})')
	
	# Adjust color scaling periodically
	if frame % 10 == 0:
		im1.set_clim(ground.min(), ground.max())
		im2.set_clim(sediment.min(), sediment.max())
		im3.set_clim(flow_vel.min(), flow_vel.max())
	
	return [im1, im2, im3]

# Create animation with adjusted interval
anim = FuncAnimation(fig, animate, frames=duration, 
					interval=200,  # Slower animation for better visualization
					blit=False)    # Disable blit for proper updates

plt.tight_layout()
plt.show()










