@echo off
chcp 65001 > nul

for /f "tokens=1-6 delims=/-: " %%a in ("%date% %time%") do (
    set YYYY=%%a
    set MM=%%b
    set DD=%%c
    set HH=%%d
    set MIN=%%e
    set SEC=%%f
)

set MSG=%YYYY%/%MM%/%DD%/%HH%/%MIN%/%SEC%

git add .
git commit -m "%MSG%"
git push origin main

echo.
echo Done. [%MSG%]
pause
