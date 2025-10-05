# Create Desktop Shortcut for Windows
# Run this script to create a desktop shortcut

$WshShell = New-Object -ComObject WScript.Shell
$DesktopPath = [System.Environment]::GetFolderPath('Desktop')
$ShortcutPath = Join-Path $DesktopPath "Multimodal RAG Chatbot.lnk"
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)

# Get current directory
$CurrentDir = Get-Location

# Set shortcut properties
$Shortcut.TargetPath = "cmd.exe"
$Shortcut.Arguments = "/c `"cd /d `"$CurrentDir`" && run_windows.bat`""
$Shortcut.WorkingDirectory = $CurrentDir
$Shortcut.Description = "Multimodal RAG Chatbot with DSPy"
$Shortcut.IconLocation = "shell32.dll,13"  # Robot icon

# Save the shortcut
$Shortcut.Save()

Write-Host "Desktop shortcut created successfully!" -ForegroundColor Green
Write-Host "Location: $ShortcutPath" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now double-click the shortcut on your desktop to start the chatbot." -ForegroundColor Yellow
Pause
