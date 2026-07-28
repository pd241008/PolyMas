package polymas.ingestion

import com.typesafe.scalalogging.StrictLogging
import io.grpc.stub.StreamObserver

class IngestionServiceImpl(
    gwasClient: GwasCatalogClient,
    immportClient: ImmPortClient,
) extends StrictLogging:

  val definition: ServerServiceDefinition =
    // TODO: Replace with auto-generated proto binding after protoc codegen
    //   IngestionServiceGrpc.bindService(new Handler, ???)
    sys.error("Protobuf stubs not yet generated — run `make proto` first")

  // TODO: Move inside proto-generated Handler once codegen is wired
  class Handler:
    def pullGwasCatalog(
        request: PullGwasRequest,
        responseObserver: StreamObserver[RawGwasPayload],
    ): Unit =
      logger.info("PullGwasCatalog called with loci: {}", request.locusIds.toSeq.mkString(", "))
      gwasClient.streamResults(request, responseObserver)

    def pullImmPort(
        request: PullImmPortRequest,
        responseObserver: StreamObserver[RawImmPortPayload],
    ): Unit =
      logger.info("PullImmPort called with studies: {}", request.studyIds.toSeq.mkString(", "))
      immportClient.streamResults(request, responseObserver)

    def healthCheck(
        request: HealthCheckRequest,
        responseObserver: StreamObserver[HealthCheckResponse],
    ): Unit =
      responseObserver.onNext(
        HealthCheckResponse(healthy = true, serviceName = "polymas-ingestion", version = "0.1.0")
      )
      responseObserver.onCompleted()
