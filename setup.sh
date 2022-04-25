echo "Setting up..."

echo "Cloning project into anti-spoofing-rPPG"

git clone "https://github.com/mhannani/anti-spoofing-rPPG"

echo "getting into anti-spoofing-rPPG"
cd anti-spoofing-rPPG

echo "Creating conda environment..."
conda create --name anti-spoofing-rPPG

echo "activating environment..."
conda activate anti-spoofing-rPPG

echo "Installing dependencies..."
conda install --file requirements.txt

echo "Done !"