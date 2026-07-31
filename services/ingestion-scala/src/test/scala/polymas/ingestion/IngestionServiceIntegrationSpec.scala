package polymas.ingestion

import com.polymas.proto.v1.{HealthCheckRequest, IngestionServiceGrpc}
import io.grpc.{Server, ServerBuilder, ManagedChannel, ManagedChannelBuilder}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class IngestionServiceIntegrationSpec
    extends AnyFlatSpec
    with Matchers
    with BeforeAndAfterAll:

  private var server: Server = null
  private var channel: ManagedChannel = null

  private val port = 50055

  override protected def beforeAll(): Unit =
    val gwasClient    = new GwasCatalogClient()
    val immportClient = new ImmPortClient()
    val service       = new IngestionServiceImpl(gwasClient, immportClient)

    server = ServerBuilder
      .forPort(port)
      .addService(service.definition)
      .build()
      .start()

    channel = ManagedChannelBuilder
      .forAddress("localhost", port)
      .usePlaintext()
      .build()

  override protected def afterAll(): Unit =
    if server != null then
      server.shutdownNow()
      server.awaitTermination(5, java.util.concurrent.TimeUnit.SECONDS)
    if channel != null then
      channel.shutdownNow()
      channel.awaitTermination(5, java.util.concurrent.TimeUnit.SECONDS)

  "IngestionService gRPC server" should "respond to health check" in {
    val stub     = IngestionServiceGrpc.newBlockingStub(channel)
    val response = stub.healthCheck(HealthCheckRequest.getDefaultInstance)

    response.getServiceName shouldBe "polymas-ingestion"
    response.getVersion shouldBe "0.1.0"
    response.getHealthy shouldBe true
  }
