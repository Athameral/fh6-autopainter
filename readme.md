# FH6-AutoPainter

> Approximate any image using primitive shapes and import them into *Forza Horizon 6*.

[中文文档 (Chinese Documentation)](readme_zh.md)

---

## 💡 About

**FH6-AutoPainter** is a lightweight utility designed to bypass the restriction in the *Forza Horizon* series that prevents users from freely importing external PNG decals. By converting target images into optimized geometric primitives (shapes), this tool automates the creation process so you can easily bring custom artwork into the game.

---

## ✨ Key Features

- **Single-File Portable Executable**: Zero complicated installation required; just download and run.
- **Ultra-Lightweight**: Only a ~3MB (after upx) executable, fully self-contained.
- **Cross-GPU Compatibility**: Works on any hardware supporting **Vulkan**, regardless of your GPU brand.

---

## 🎬 Demos

### 1. Generation Process
*Watch how the algorithm approximates a source image using primitive shapes:*
> ![Generation Demo](https://github.com/user-attachments/assets/004685a6-7536-439c-9df9-6f4af289079e)

### 2. In-Game Import
*See how the generated layout translates into the FH6 vinyl editor:*
> ![Import Demo](https://github.com/user-attachments/assets/6cd0a310-1689-4398-87a5-b6ec6002393a)

---

## 🚀 Usage Instructions

1. Download the latest pre-compiled `FH6-AutoPainter.exe` from the Release page.
2. Launch the application.
3. **Drag in** your target image (PNG/JPG).
4. Configure the number of shapes and other parameters according to your preference. **Make sure every occuranve of the number of shapes is consistent.**
5. Follow the in-game injection guidelines to load it into *Forza Horizon 6*.

---

## 🛠️ Build Instructions

If you prefer to build the project from source, follow the steps below:

### Prerequisites
- A modern C++ compiler toolchain (e.g., `llvm-mingw` or MSVC).
- CMake (version 3.20 or higher).
- Git.
 
> Note: MSVC is the default compiler used by `taichi` on Windows platform, which should usually just work, though not tested. However, to ensure this project can be built without downloading additional dependencies, it is recommended to use `llvm-mingw` as the compiler toolchain.

TODO