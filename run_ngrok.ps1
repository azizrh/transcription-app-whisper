# Define download URL and target directory
$ngrokUrl = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip"
$ngrokZip = "ngrok.zip"
$ngrokExe = "ngrok.exe"
 
# Create directory if it doesn't exist
$targetDir = "$env:USERPROFILE\ngrok"
if (-not (Test-Path $targetDir)) {
    New-Item -ItemType Directory -Path $targetDir
}
 
# Download ngrok
Write-Host "Downloading ngrok..."
Invoke-WebRequest -Uri $ngrokUrl -OutFile "$targetDir\$ngrokZip"
 
# Extract zip
Write-Host "Extracting ngrok..."
Expand-Archive -Path "$targetDir\$ngrokZip" -DestinationPath $targetDir -Force
 
# Clean up zip file
Remove-Item "$targetDir\$ngrokZip"
 
# Change to ngrok directory
Set-Location $targetDir
 
# Start ngrok
Write-Host "Starting ngrok on port 5000..."
.\ngrok http 5000