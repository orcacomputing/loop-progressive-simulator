FROM python:3.10-slim as base

# Install system dependencies, clean up in one RUN to reduce image size
RUN apt-get update && \
    apt-get install -y python3-pip python3-packaging git htop && \
    rm -rf /var/lib/apt/lists/*  # Clean up to reduce layer size

# Upgrade pip and install basic Python tools
RUN pip3 install --no-cache-dir -U pip setuptools wheel

# Copy the Python dependencies file to the container

COPY requirements.txt .
COPY loop_progressive_simulator/bscircuits.py loop_progressive_simulator/
COPY loop_progressive_simulator/complexity_analysis.py loop_progressive_simulator/
COPY loop_progressive_simulator/complexity_experiment.py loop_progressive_simulator/
COPY loop_progressive_simulator/factorialtable_nosym.py loop_progressive_simulator/
COPY loop_progressive_simulator/fock_states.py loop_progressive_simulator/
COPY loop_progressive_simulator/lattice_paths.py loop_progressive_simulator/
COPY loop_progressive_simulator/number_basis.py loop_progressive_simulator/
COPY loop_progressive_simulator/run_experiment_families.py loop_progressive_simulator/
COPY loop_progressive_simulator/step_simulator_nosym.py loop_progressive_simulator/
COPY loop_progressive_simulator/utils.py loop_progressive_simulator/

# Install Python dependencies from the requirements file
RUN pip3 install --no-cache-dir -r requirements.txt

# Set environment variables to prevent Python from generating .pyc files and to turn off buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install the IPython kernel
RUN pip3 install --no-cache-dir ipykernel

# Start a new build stage for development-specific settings
FROM base as dev

# Set the default command to open a Bash shell
CMD ["bash"]
