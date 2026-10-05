@echo off
cd /d "C:\Users\antdu953\OneDrive - Uppsala universitet (1)\wec_modeling_benchmark\docs"
call conda activate surrogate
start mkdocs serve
timeout /t 3 /nobreak
start http://127.0.0.1:8000