package api

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Server HTTP 服务器
type Server struct {
	srv          *http.Server
	handler      *Handler
	trainHandler *TrainHandler
}

// NewServer 创建 HTTP 服务器
func NewServer(port int, h *Handler, th *TrainHandler) *Server {
	mux := http.NewServeMux()
	s := &Server{handler: h, trainHandler: th}
	s.registerRoutes(mux)

	s.srv = &http.Server{
		Addr:         fmt.Sprintf(":%d", port),
		Handler:      mux,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  60 * time.Second,
	}
	return s
}

func (s *Server) registerRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/v1/predict/ctr", s.handler.HandlePredictCTR)
	mux.HandleFunc("/v1/predict/batch", s.handler.HandlePredictBatch)
	mux.HandleFunc("/v1/model/current", s.handler.HandleModelCurrent)
	mux.HandleFunc("/v1/model/reload", s.handler.HandleModelReload)
	mux.HandleFunc("/v1/models", s.handler.HandleListModels)
	mux.HandleFunc("/v1/train_sample", s.trainHandler.HandleTrainSample)
	mux.HandleFunc("/health", s.handler.HandleHealth)
	mux.Handle("/metrics", promhttp.Handler())
}

// Start 启动服务器（阻塞）
func (s *Server) Start() error {
	return s.srv.ListenAndServe()
}

// Shutdown 优雅停机
func (s *Server) Shutdown(ctx context.Context) error {
	return s.srv.Shutdown(ctx)
}
