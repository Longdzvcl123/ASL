# .----------------.  .----------------.  .----------------.  .----------------.  .----------------.  .----------------. 
#| .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
#| |  _______     | || |  _________   | || |      __      | || |  ________    | || | ____    ____ | || |  _________   | |
#| | |_   __ \    | || | |_   ___  |  | || |     /  \     | || | |_   ___ `.  | || ||_   \  /   _|| || | |_   ___  |  | |
#| |   | |__) |   | || |   | |_  \_|  | || |    / /\ \    | || |   | |   `. \ | || |  |   \/   |  | || |   | |_  \_|  | |
#| |   |  __ /    | || |   |  _|  _   | || |   / ____ \   | || |   | |    | | | || |  | |\  /| |  | || |   |  _|  _   | |
#| |  _| |  \ \_  | || |  _| |___/ |  | || | _/ /    \ \_ | || |  _| |___.' / | || | _| |_\/_| |_ | || |  _| |___/ |  | |
#| | |____| |___| | || | |_________|  | || ||____|  |____|| || | |________.'  | || ||_____||_____|| || | |_________|  | |
#| |              | || |              | || |              | || |              | || |              | || |              | |
#| '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
# '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------' 



# THIS IS FOR WINDOWS IF YOU ARE ON LINUX SKIP DIRECTLY TO STEP 6


# 1) In case the activation of the venv does not work
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 2) Create the virtual environnement
python -m venv venv_asl

# 3) Activate the virtual environnement
.\venv_asl\Scripts\activate

# 4) Write this command in the terminal, if it returns 32 continue else go to point (6)
python -c "import struct; print(struct.calcsize('P') * 8)"

# 5) Installation of the Torch Library ==> if 32
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 6) Installation of the Torch Library ==> if 64
pip install torch torchvision torchaudio

# 7) Installation of MediaPipe, OpenCV and NumPy
pip install mediapipe opencv-python numpy

# 8) Make sure that every package installed is up to date
python.exe -m pip install --upgrade pip

# 9) Navigate to the safe directory
cd ./safe

# 10) Launch the UI and try it !
python demo.py