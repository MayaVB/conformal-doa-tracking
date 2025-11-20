# Dockerfile for Dataset Generation Environment
# Based on NVIDIA PyTorch container

FROM nvcr.io/nvidia/pytorch:22.11-py3

# Set arguments for user creation
ARG USERNAME=mayavb
ARG UID=30772
ARG GID=2102

# Set environment variables
ENV DISPLAY=:0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV MATLAB_WORKING_DIR=/src

# Create user and setup directories
RUN groupadd -g $GID $USERNAME || true && \
    useradd -m -u $UID -g $GID -s /bin/bash $USERNAME || true && \
    mkdir -p /dev/shm && chmod 1777 /dev/shm && \
    mkdir -p /home/$USERNAME/Documents/MATLAB && \
    chown -R $USERNAME:$GID /home/$USERNAME/Documents/ && \
    chown -R $USERNAME:$GID /home/$USERNAME

# Set working directory
WORKDIR /workspace/unet

# Copy requirements file
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . /workspace/unet/

# Change ownership of workspace
RUN chown -R $USERNAME:$GID /workspace/unet

# Switch to user
USER $USERNAME

# Set default command
CMD ["/bin/bash"]