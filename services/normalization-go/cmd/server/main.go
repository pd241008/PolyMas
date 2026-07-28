package main

import (
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"

	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"

	"github.com/polymas/normalization-go/internal/handler"
)

func main() {
	port := os.Getenv("GRPC_PORT")
	if port == "" {
		port = "50052"
	}

	addr := fmt.Sprintf(":%s", port)
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		log.Fatalf("failed to listen on %s: %v", addr, err)
	}

	s := grpc.NewServer()

	// Register normalization service
	normalizationSvc := handler.NewNormalizationService()
	// TODO: pb.RegisterNormalizationServiceServer(s, normalizationSvc)

	// Register health check
	healthSrv := health.NewServer()
	healthpb.RegisterHealthServer(s, healthSrv)
	healthSrv.SetServingStatus("polymas.v1.NormalizationService", healthpb.HealthCheckResponse_SERVING)

	// Graceful shutdown
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		log.Println("shutting down normalization service...")
		healthSrv.SetServingStatus("polymas.v1.NormalizationService", healthpb.HealthCheckResponse_NOT_SERVING)
		s.GracefulStop()
	}()

	log.Printf("normalization service listening on %s", addr)
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
