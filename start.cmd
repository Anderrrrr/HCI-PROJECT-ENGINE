@echo off
setlocal
set "BUNDLED_PNPM=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\bin\fallback\pnpm.cmd"

where pnpm >nul 2>nul
if %ERRORLEVEL% EQU 0 (
  pnpm exec vite --host 127.0.0.1
  exit /b %ERRORLEVEL%
)

if exist "%BUNDLED_PNPM%" (
  call "%BUNDLED_PNPM%" exec vite --host 127.0.0.1
  exit /b %ERRORLEVEL%
)

echo Could not find pnpm. Install Node.js with Corepack or run this project from Codex.
pause
exit /b 1
