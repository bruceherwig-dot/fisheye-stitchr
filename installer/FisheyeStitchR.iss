; Inno Setup script for Fisheye StitchR. Compiled in CI by ISCC.exe with /DAppVersion=<version>.
; A one-click Setup.exe is far kinder than a raw zip of thousands of small files.

#ifndef AppVersion
  #define AppVersion "0.0"
#endif

#define AppName "Fisheye StitchR"
#define AppPublisher "Bruce Herwig"
#define AppURL "https://github.com/bruceherwig-dot/fisheye-stitchr"
#define AppExeName "FisheyeStitchR.exe"

[Setup]
; Stable AppId: lets later installers recognize an existing install and upgrade it in place. Never change it.
AppId={{3F6B9C1E-5D24-4A7B-8E31-C0A94D7E2B58}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
DefaultDirName={localappdata}\Programs\FisheyeStitchR
DefaultGroupName=Fisheye StitchR
DisableProgramGroupPage=yes
OutputDir=.
OutputBaseFilename=FisheyeStitchRSetup
SetupIconFile=..\assets\icon.ico
Compression=lzma2/ultra
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
Source: "..\dist\FisheyeStitchR\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Fisheye StitchR"; Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall Fisheye StitchR"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Fisheye StitchR"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch Fisheye StitchR"; Flags: nowait postinstall skipifsilent
