"""Entry point for the ML Engine gRPC server."""

from __future__ import annotations

import logging
import sys

from polymas_ml.serving.grpc_server import MLEngineServicer

logger = logging.getLogger("polymas_ml")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    port = 50054

    logger.info("Polymas ML Engine starting on port %d", port)

    # TODO: Wire up grpc.server with auto-generated stubs
    # from concurrent import futures
    # server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    # add_MLEngineServiceServicer_to_server(MLEngineServicer(), server)
    # server.add_insecure_port(f"[::]:{port}")
    # server.start()
    # server.wait_for_termination()

    MLEngineServicer()
    logger.info("ML Engine servicer initialized. gRPC server pending protobuf codegen.")

    # Placeholder: keep alive
    try:
        import signal
        signal.pause()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down ML Engine")


if __name__ == "__main__":
    main()
