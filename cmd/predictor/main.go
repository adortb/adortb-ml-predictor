package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/adortb/adortb-ml-predictor/internal/api"
	"github.com/adortb/adortb-ml-predictor/internal/model"
	"github.com/adortb/adortb-ml-predictor/internal/prediction"
	"github.com/redis/go-redis/v9"
)

func main() {
	cfg := loadConfig()

	// 初始化 Redis（可选）
	var rdb *redis.Client
	if cfg.redisAddr != "" {
		rdb = redis.NewClient(&redis.Options{
			Addr:         cfg.redisAddr,
			DialTimeout:  2 * time.Second,
			ReadTimeout:  1 * time.Second,
			WriteTimeout: 1 * time.Second,
		})
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		if err := rdb.Ping(ctx).Err(); err != nil {
			log.Printf("[WARN] Redis 连接失败，仅使用本地模型: %v", err)
			rdb = nil
		}
	}

	// 初始化模型注册表
	registry, err := model.NewRegistry(cfg.modelPath, rdb)
	if err != nil {
		log.Fatalf("初始化模型失败: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	registry.StartWatcher(ctx)

	// 初始化预测流水线
	pipeline := prediction.NewPipeline(registry)

	// 初始化 FTRL 在线学习模型
	ftrlModel := model.NewFTRLModel(model.DefaultFTRLConfig())

	// 初始化 HTTP 服务
	handler := api.NewHandler(pipeline, registry, ftrlModel)
	trainHandler := api.NewTrainHandler(ftrlModel)
	srv := api.NewServer(cfg.port, handler, trainHandler)

	// 优雅停机
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		log.Printf("adortb-ml-predictor 启动，端口 :%d，模型 %s", cfg.port, registry.Current().GetVersion())
		if err := srv.Start(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("服务器错误: %v", err)
		}
	}()

	<-sigCh
	log.Println("收到停机信号，开始优雅退出...")

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()

	registry.Stop()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("停机错误: %v", err)
	}
	log.Println("服务已停止")
}

type config struct {
	port      int
	modelPath string
	redisAddr string
}

func loadConfig() config {
	port := 8087
	if v := os.Getenv("PREDICTOR_PORT"); v != "" {
		fmt.Sscanf(v, "%d", &port)
	}

	modelPath := os.Getenv("MODEL_PATH")
	if modelPath == "" {
		modelPath = "models/lr_ctr_v1.json"
	}

	return config{
		port:      port,
		modelPath: modelPath,
		redisAddr: os.Getenv("REDIS_ADDR"),
	}
}
