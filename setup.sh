echo "Setting up..."

echo "Cloning project into anti-spoofing-rPPG"

git clone "https://github.com/mhannani/anti-spoofing-rPPG"

echo "getting into anti_spoofing_rPPG"
cd anti-spoofing-rPPG

echo "Creating conda environment..."
conda create --name anti_spoofing_rPPG

echo "activating environment..."
conda activate anti_spoofing_rPPG

echo "Installing dependencies..."
conda install --file requirements.txt

echo "Done !"