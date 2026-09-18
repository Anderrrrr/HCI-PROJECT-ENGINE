@echo off
setlocal
set "BUNDLED_PNPM=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\bin\fallback\pnpm.cmd"
set "BUNDLED_NODE_DIR=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin"

where pnpm >nul 2>nul
if errorlevel 1 goto bundled
where node >nul 2>nul
if errorlevel 1 goto bundled
pnpm run dev
exit /b %ERRORLEVEL%

:bundled
if exist "%BUNDLED_PNPM%" if exist "%BUNDLED_NODE_DIR%\node.exe" (
  set "PATH=%BUNDLED_NODE_DIR%;%PATH%"
  call "%BUNDLED_PNPM%" run dev
  exit /b %ERRORLEVEL%
)

echo Could not find Node.js and pnpm.
echo Install Node.js with Corepack, then run: pnpm install ^&^& pnpm dev
pause
exit /b 1
