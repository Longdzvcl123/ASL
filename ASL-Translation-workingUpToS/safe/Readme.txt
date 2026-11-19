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


# Requirements: 64-bit Python 3.10 or newer (PyTorch does NOT exist for 32-bit Python)
# Check:  python -c "import struct, sys; print(struct.calcsize('P') * 8, sys.version)"   --> must print 64


# Full documentation: see README.md at the root of the repository

# ===================== INSTALLATION (Windows; on Linux/macOS skip step 1, use source venv_asl/bin/activate) =====================

# 1) In case the activation of the venv does not work
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 2) Create and activate the virtual environment
python -m venv venv_asl
.\venv_asl\Scripts\activate

# 3) Update pip, then install everything
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4) (Optional) NVIDIA GPU: reinstall torch with the CUDA command given on https://pytorch.org/get-started/locally/ , e.g.
pip install torch --index-url https://download.pytorch.org/whl/cu121


# ===================== USAGE (run every command from the safe directory) =====================
cd ./safe

# Try the demo with the provided model (press 'q' to quit)
python demo.py

# Full pipeline to (re)train with your own data:
python dataset_fill_fixed.py --csv my_data.csv     # 1. record letters with the webcam
python pickle_data.py --csv my_data.csv            # 2. normalize + split into New_asl_data.pkl
python check_data.py                               # 3. (optional) check classes / counts
python train.py                                    # 4. train -> best_asl_model.pt (prints test accuracy)
python demo.py                                     # 5. test it live
