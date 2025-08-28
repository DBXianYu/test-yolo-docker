# 工业缺陷检测项目 Makefile
# 一键构建、运行、测试

.PHONY: help build run stop clean test logs shell

# 默认目标
help:
	@echo "🚀 工业缺陷检测 YOLOv8 部署脚本"
	@echo ""
	@echo "可用命令："
	@echo "  make build     - 构建Docker镜像"
	@echo "  make run       - 启动服务容器"
	@echo "  make stop      - 停止容器"
	@echo "  make test      - 测试API接口"
	@echo "  make logs      - 查看容器日志"
	@echo "  make shell     - 进入容器shell"
	@echo "  make clean     - 清理容器和镜像"
	@echo "  make size      - 查看镜像大小"
	@echo ""
	@echo "🎯 目标：镜像 < 1GB，启动 < 30s，推理 < 100ms"

# 配置变量
IMAGE_NAME=defect-detection
CONTAINER_NAME=defect-api
PORT=8000

# 构建镜像
build:
	@echo "🔨 构建Docker镜像..."
	docker build -t $(IMAGE_NAME):latest .
	@echo "✅ 构建完成！"
	@make size

# 启动服务
run:
	@echo "🚀 启动缺陷检测服务..."
	docker run -d \
		--name $(CONTAINER_NAME) \
		--gpus all \
		-p $(PORT):8000 \
		--restart unless-stopped \
		$(IMAGE_NAME):latest
	@echo "✅ 服务已启动！"
	@echo "📡 API地址: http://localhost:$(PORT)"
	@echo "📖 文档地址: http://localhost:$(PORT)/docs"
	@echo "❤️  健康检查: http://localhost:$(PORT)/health"

# 停止容器
stop:
	@echo "🛑 停止服务..."
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true
	@echo "✅ 服务已停止！"

# 查看日志
logs:
	docker logs -f $(CONTAINER_NAME)

# 进入容器
shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash

# 测试API
test:
	@echo "🧪 测试API接口..."
	@echo "1. 健康检查:"
	curl -s http://localhost:$(PORT)/health | python -m json.tool || echo "❌ 健康检查失败"
	@echo "\n2. 根接口:"
	curl -s http://localhost:$(PORT)/ | python -m json.tool || echo "❌ 根接口测试失败"
	@echo "\n✅ 基础测试完成！"
	@echo "📖 完整API文档: http://localhost:$(PORT)/docs"

# 压力测试（需要先准备测试图片）
stress-test:
	@echo "⚡ 压力测试（需要test.jpg）..."
	@for i in {1..10}; do \
		echo "第$$i次请求:"; \
		time curl -X POST "http://localhost:$(PORT)/predict" \
		-H "accept: application/json" \
		-H "Content-Type: multipart/form-data" \
		-F "file=@test.jpg" | python -m json.tool; \
		echo ""; \
	done

# 查看镜像大小
size:
	@echo "📏 Docker镜像大小:"
	docker images $(IMAGE_NAME):latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

# 清理所有相关资源
clean:
	@echo "🧹 清理Docker资源..."
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true
	docker rmi $(IMAGE_NAME):latest || true
	docker system prune -f
	@echo "✅ 清理完成！"

# 开发模式（本地运行，不用Docker）
dev:
	@echo "💻 开发模式启动..."
	pip install -r requirements.txt
	uvicorn main:app --host 0.0.0.0 --port $(PORT) --reload

# 生产环境部署
deploy:
	@echo "🌟 生产环境部署..."
	@make build
	@make stop
	@make run
	@sleep 10
	@make test
	@echo "🎉 部署完成！服务地址: http://localhost:$(PORT)"

# 性能监控
monitor:
	@echo "📊 性能监控..."
	@while true; do \
		echo "=== $(shell date) ==="; \
		docker stats $(CONTAINER_NAME) --no-stream; \
		curl -s http://localhost:$(PORT)/metrics; \
		echo ""; \
		sleep 5; \
	done
