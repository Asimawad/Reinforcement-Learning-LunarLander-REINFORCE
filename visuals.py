import time
import numpy as np
import pickle
import matplotlib.pyplot as plt

with open("./learned_params.pkl", "rb") as param_file:
    loaded_params = pickle.load(param_file)

with open("./episode_returns.pkl", "rb") as file:
    loaded_episode_returns = pickle.load(file)
# print(loaded_episode_returns)
plt.figure(figsize=(10, 6))  # Adjust figure size for better visibility
x = [int(x) for x in loaded_episode_returns]
plt.plot(range(len(x)), x)  # Corrected the plot call
plt.savefig("policy.png")


# # import vlc

# # Path to your video file
# video_path = "video/q_learning/eval/rl-video-episode-792.mp4"

# # Create VLC instance
# player = vlc.MediaPlayer(video_path)

# # Play the video
# player.play()

# # Wait for the video to finish (assuming it's 10 seconds long)
# time.sleep(10)

# # Stop the player (optional)
# player.stop()
