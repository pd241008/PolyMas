package polymas.ingestion

import com.typesafe.scalalogging.StrictLogging
import io.grpc.{ServerBuilder, ServerServiceDefinition}

object IngestionServer extends StrictLogging:

  private val Port = sys.env.getOrElse("GRPC_PORT", "50051").toInt

  def main(args: Array[String]): Unit =
    logger.info("Starting Polymas Ingestion Service on port {}", Port)

    val gwasClient    = new GwasCatalogClient()
    val immportClient = new ImmPortClient()
    val service       = new IngestionServiceImpl(gwasClient, immportClient)

    val server = ServerBuilder
      .forPort(Port)
      .addService(service.definition)
      .build()
      .start()

    logger.info("Ingestion Service listening on port {}", Port)

    sys.addShutdownHook:
      logger.info("Shutting down Ingestion Service...")
      server.shutdown()

    server.awaitTermination()
