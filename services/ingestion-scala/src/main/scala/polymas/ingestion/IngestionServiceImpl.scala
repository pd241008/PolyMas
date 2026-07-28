package polymas.ingestion

import com.polymas.proto.v1.*
import com.typesafe.scalalogging.StrictLogging
import io.grpc.ServerServiceDefinition
import io.grpc.stub.StreamObserver

class IngestionServiceImpl(
    gwasClient: GwasCatalogClient,
    immportClient: ImmPortClient,
) extends StrictLogging:

  val definition: ServerServiceDefinition =
    IngestionServiceGrpc.bindService(new IngestionServiceHandler)

  class IngestionServiceHandler extends IngestionServiceGrpc.IngestionServiceImplBase:
    override def pullGwasCatalog(
        request: PullGwasRequest,
        responseObserver: StreamObserver[RawGwasPayload],
    ): Unit =
      logger.info("PullGwasCatalog called with loci: {}", request.getLocusIdsList)
      gwasClient.streamResults(request, responseObserver)

    override def pullImmPort(
        request: PullImmPortRequest,
        responseObserver: StreamObserver[RawImmPortPayload],
    ): Unit =
      logger.info("PullImmPort called with studies: {}", request.getStudyIdsList)
      immportClient.streamResults(request, responseObserver)

    override def healthCheck(
        request: HealthCheckRequest,
        responseObserver: StreamObserver[HealthCheckResponse],
    ): Unit =
      val response = HealthCheckResponse.newBuilder()
        .setHealthy(true)
        .setServiceName("polymas-ingestion")
        .setVersion("0.1.0")
        .build()
      responseObserver.onNext(response)
      responseObserver.onCompleted()
