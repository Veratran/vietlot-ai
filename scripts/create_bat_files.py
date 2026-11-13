import pathlib

ROOT = pathlib.Path("D:/vietlot-ai")

# 1. run_fields_quant.bat
fields_quant_bat = """@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ===============================================================================
echo  VIETLOT AI - CHAY DU DOAN FIELDS-QUANT (POWER 6/55 VA MEGA 6/45)
echo ===============================================================================
echo.
echo [1/2] Dang tinh toan he phu 12 so, loc phan dam dong va khu nhieu RMT...
python scripts\\build_predictions.py --game power655,power645 --wheel 12 --anti-crowd --rmt --mc-simulations 200

echo.
echo [2/2] Dang cap nhat toan bo API JSON tinh...
python scripts\\build_api.py

echo.
echo ===============================================================================
echo  HOAN TAT! Du lieu da duoc luu vao data\\predictions.jsonl va api\\
echo ===============================================================================
pause
"""

# 2. run_tests.bat
tests_bat = """@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ===============================================================================
echo  VIETLOT AI - CHAY TOAN BO BO KIEM THU TU DONG (408 TESTS)
echo ===============================================================================
echo.
python -m pytest tests -v
echo.
echo ===============================================================================
echo  Kiem thu hoan tat!
echo ===============================================================================
pause
"""

# 3. run_predictions.bat (Master interactive menu)
predictions_bat = """@echo off
chcp 65001 >nul
cd /d "%~dp0"

:MENU
cls
echo ===============================================================================
echo            VIETLOT AI - HE THONG DU DOAN VA QUAN TRI DINH LUONG
echo ===============================================================================
echo.
echo  [1] Chay Du Doan Toan Bo Game (Standard Pipeline)
echo  [2] Chay Du Doan Dinh Cao Fields-Quant (Dan 12 ve, Phan dam dong, Loc RMT)
echo  [3] Chay Du Doan Che Do Arbitrage (Chi mua khi EV ^> 0, Jackpot ^> 230 ty)
echo  [4] Xay Dung Tang API Tinh (Sinh tat ca endpoint JSON api\\*.json)
echo  [5] Chay Cap Nhat 1-Click (Du Doan Nang Cao + Build API + Mo Localhost)
echo  [6] Chay Kiem Thu Toan Dien He Thong (Pytest 408 Tests)
echo  [7] Khoi Dong Local Web Server (http://localhost:8000)
echo  [8] Mo Truc Tiep File Du Doan (data\\predictions.jsonl)
echo  [0] Thoat
echo.
echo ===============================================================================
set /p opt="Vui long chon chuc nang [0-8]: "

if "%opt%"=="1" goto OP_STANDARD
if "%opt%"=="2" goto OP_FIELDS_QUANT
if "%opt%"=="3" goto OP_ARBITRAGE
if "%opt%"=="4" goto OP_BUILD_API
if "%opt%"=="5" goto OP_ONE_CLICK
if "%opt%"=="6" goto OP_TEST_ALL
if "%opt%"=="7" goto OP_SERVER
if "%opt%"=="8" goto OP_OPEN_PRED
if "%opt%"=="0" goto EXIT
goto MENU

:OP_STANDARD
cls
echo [1] Dang chay sinh du doan tieu chuan cho tat ca cac game...
python scripts\\build_predictions.py
echo.
echo Hoan tat! Dang tu dong cap nhat API...
python scripts\\build_api.py
echo.
pause
goto MENU

:OP_FIELDS_QUANT
cls
echo [2] Dang chay du doan Nang cao Fields-Quant:
echo     - Power 6/55: Dan bao rut gon 12 so (14 ve) + Loc Phan dam dong + Khu nhieu RMT
echo     - Mega 6/45 : Dan bao rut gon 12 so (14 ve) + Loc Phan dam dong + Khu nhieu RMT
echo.
python scripts\\build_predictions.py --game power655 --wheel 12 --anti-crowd --rmt --mc-simulations 200
python scripts\\build_predictions.py --game power645 --wheel 12 --anti-crowd --rmt --mc-simulations 200
echo.
echo Hoan tat! Dang tu dong cap nhat API...
python scripts\\build_api.py
echo.
pause
goto MENU

:OP_ARBITRAGE
cls
echo [3] Dang chay du doan theo che do Arbitrage nghiem ngat:
echo     Chi cho phep mua khi ky vong toan hoc EV ^> 0 (Power 6/55 ^> 230 ty, Mega 6/45 ^> 60 ty).
echo     Neu chua du dieu kien, FSM chuyen trang thai an toan DORMANT_SKIP.
echo.
python scripts\\build_predictions.py --game power655,power645 --arbitrage-only --wheel 12 --anti-crowd
echo.
echo Hoan tat! Dang tu dong cap nhat API...
python scripts\\build_api.py
echo.
pause
goto MENU

:OP_BUILD_API
cls
echo [4] Dang xay dung toan bo tang API JSON tinh...
python scripts\\build_api.py
echo.
pause
goto MENU

:OP_ONE_CLICK
cls
echo [5] Chay Cap Nhat 1-Click:
echo     Buoc 1: Chay mo hinh du doan dinh luong Fields-Quant...
python scripts\\build_predictions.py --wheel 12 --anti-crowd --rmt --mc-simulations 200
echo.
echo     Buoc 2: Xuat ban toan bo API JSON tinh...
python scripts\\build_api.py
echo.
echo     Buoc 3: Mo trinh duyet xem trang web...
start "" "http://localhost:8000"
echo.
echo Dang mo Local Server tai cong 8000 (nhan Ctrl+C de dung server khi xong)...
python -m http.server 8000
goto MENU

:OP_TEST_ALL
cls
echo [6] Dang chay toan bo bo kiem thu tu dong (408 tests)...
python -m pytest tests -v
echo.
pause
goto MENU

:OP_SERVER
cls
echo [7] Dang khoi dong may chu Web cuc bo tai http://localhost:8000...
start "" "http://localhost:8000"
python -m http.server 8000
goto MENU

:OP_OPEN_PRED
cls
if exist "data\\predictions.jsonl" (
    start notepad "data\\predictions.jsonl"
) else (
    echo Chua tim thay data\\predictions.jsonl. Hay chay chuc nang [1] hoac [2] truoc!
)
pause
goto MENU

:EXIT
echo Tam biet!
exit /b 0
"""

def save_crlf(path, text):
    crlf = text.replace("\r\n", "\n").replace("\n", "\r\n")
    path.write_bytes(crlf.encode("utf-8"))
    print(f"Saved: {path.name} ({len(crlf)} chars, CRLF)")

save_crlf(ROOT / "run_fields_quant.bat", fields_quant_bat)
save_crlf(ROOT / "run_tests.bat", tests_bat)
save_crlf(ROOT / "run_predictions.bat", predictions_bat)
