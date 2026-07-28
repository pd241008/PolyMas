package polymas.ingestion

import io.grpc.stub.StreamObserver

class IngestionServiceImpl(
    gwasClient: GwasCatalogClient,
    immportClient: ImmPortClient,
) extends StrictLogging {

  import polymas.proto.v1.IngestionServiceGrpc._

  val definition: io.grpc.ServerServiceDefinition =
    bindService(new IngestionServiceHandler, io.grpc.ServerBuilder.forPort(0).build().getExecutor)

  class IngestionServiceHandler extends IngestionService {
    override def pullGwasCatalog(
      request: PullGwasRequest,
      responseObserver: StreamObserver[RawGwasPayload],
    ): Unit = {
      logger.info("PullGwasCatalog called with loci: {}", request.locusIds.asScala.mkString(", "))
      // TODO: Implement GWAS Catalog API pull with streaming backpressure
      gwasClient.streamResults(request, responseObserver)
    }

    override def pullImmPort(
      request: PullImmPortRequest,
      responseObserver: StreamObserver[RawImmPortPayload],
    ): Unit = {
      logger.info("PullImmPort called with studies: {}", request.studyIds.asScala.mkString(", "))
      // TODO: Implement ImmPort API pull with streaming backpressure
      immportClient.streamResults(request, responseObserver)
    }

    override def healthCheck(
      request: HealthCheckRequest,
      responseObserver: StreamObserver[HealthCheckResponse],
    ): Unit = {
      val response = HealthCheckResponse(
        healthy = true,
        serviceName = "polymas-ingestion",
        version = "0.1.0",
      )
      responseObserver.onNext(response)
      responseObserver.onCompleted()
    }
  }
}
