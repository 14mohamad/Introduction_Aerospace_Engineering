# Star Identifier Flutter App

A Flutter mobile application that identifies stars in astronomical images using the Flask backend server.

## Features

- Take photos using camera or select from gallery
- Adjust detection threshold with slider
- Send images to Flask server for star identification
- Display annotated results with identified stars
- Modern dark UI with space theme
- Shows star details including name, magnitude, and spectral type

## Setup

1. Install Flutter dependencies:
```bash
flutter pub get
```

2. Update the server URL in `lib/services/api_service.dart`:
```dart
static const String baseUrl = 'http://YOUR_SERVER_IP:5000';
```
Replace `YOUR_SERVER_IP` with your Flask server's IP address.

3. Run the app:
```bash
flutter run
```

## Important Configuration

### For Android
The app requires the following permissions:
- Camera
- Internet
- Storage

These are already configured in `AndroidManifest.xml`.

### For iOS
Add the following to `ios/Runner/Info.plist`:
```xml
<key>NSCameraUsageDescription</key>
<string>This app needs camera access to take photos of the night sky</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>This app needs photo library access to select star images</string>
```

## Usage

1. Launch the app
2. Tap "Select Image" to choose camera or gallery
3. Adjust the threshold slider if needed (default: 120)
4. Tap "Identify Stars" to process the image
5. View results showing identified stars with their details

## Server Requirements

Make sure the Flask server is running and accessible from your device:
```bash
python flask_star_identifier.py
```

The server should be running on port 5000 and accessible from your mobile device's network.