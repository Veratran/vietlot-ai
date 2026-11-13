@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ===============================================================================
echo  VIETLOT AI - CHAY DU DOAN FIELDS-QUANT (POWER 6/55 VA MEGA 6/45)
echo ===============================================================================
echo.
echo [1/2] Dang tinh toan he phu 12 so, loc phan dam dong va khu nhieu RMT...
python scripts\build_predictions.py --game power655,power645 --wheel 12 --anti-crowd --rmt --mc-simulations 200

echo.
echo [2/2] Dang cap nhat toan bo API JSON tinh...
python scripts\build_api.py

echo.
echo ===============================================================================
echo  HOAN TAT! Du lieu da duoc luu vao data\predictions.jsonl va api\
echo ===============================================================================
pause
