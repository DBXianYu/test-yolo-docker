@echo off
:: 工业缺陷检测 Windows 一键部署脚本
:: 适配PowerShell环境

echo 🚀 工业缺陷检测 YOLOv8 部署脚本 (Windows版)
echo.

if "%1"=="" goto help
if "%1"=="build" goto build
if "%1"=="run" goto run
if "%1"=="stop" goto stop
if "%1"=="test" goto test
if "%1"=="logs" goto logs
if "%1"=="clean" goto clean
if "%1"=="size" goto size
goto help

:help
echo 可用命令:
echo   deploy.bat build     - 构建Docker镜像
echo   deploy.bat run       - 启动服务容器
echo   deploy.bat stop      - 停止容器
echo   deploy.bat test      - 测试API接口
echo   deploy.bat logs      - 查看容器日志
echo   deploy.bat clean     - 清理容器和镜像
echo   deploy.bat size      - 查看镜像大小
echo.
echo 🎯 目标: 镜像 ^< 1GB, 启动 ^< 30s, 推理 ^< 100ms
goto end

:build
echo 🔨 构建轻量Docker镜像...
docker build --no-cache -f Dockerfile.final -t defect-detection:lite .
if %errorlevel% neq 0 (
    echo ❌ 构建失败！
    goto end
)
echo ✅ 构建完成！
call :size
goto end

:run
echo 🚀 启动缺陷检测服务...
docker run -d --name defect-api --gpus all -p 8000:8000 --restart unless-stopped defect-detection:lite
if %errorlevel% neq 0 (
    echo ❌ 启动失败！
    goto end
)
echo ✅ 服务已启动！
echo 📡 API地址: http://localhost:8000
echo 📖 文档地址: http://localhost:8000/docs
echo ❤️  健康检查: http://localhost:8000/health
goto end

:stop
echo 🛑 停止服务...
docker stop defect-api 2>nul
docker rm defect-api 2>nul
echo ✅ 服务已停止！
goto end

:test
echo 🧪 测试API接口...
echo 1. 健康检查:
powershell -Command "try { $response = Invoke-RestMethod -Uri 'http://localhost:8000/health'; $response | ConvertTo-Json } catch { Write-Host '❌ 健康检查失败' }"
echo.
echo 2. 根接口:
powershell -Command "try { $response = Invoke-RestMethod -Uri 'http://localhost:8000/'; $response | ConvertTo-Json } catch { Write-Host '❌ 根接口测试失败' }"
echo.
echo ✅ 基础测试完成！
echo 📖 完整API文档: http://localhost:8000/docs
goto end

:logs
docker logs -f defect-api
goto end

:size
echo 📏 Docker镜像大小:
docker images defect-detection:lite
goto end

:clean
echo 🧹 清理Docker资源...
docker stop defect-api 2>nul
docker rm defect-api 2>nul  
docker rmi defect-detection:lite 2>nul
docker system prune -f
echo ✅ 清理完成！
goto end

:end
pause
