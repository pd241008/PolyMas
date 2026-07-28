package polymas.ingestion

import io.grpc.stub.StreamObserver

class ImmPortClient {

  private val baseUrl = sys.env.getOrElse(
    "IMMPORT_API_BASE_URL",
    "https://immport.org/immport/api",
  )

  def streamResults(
    request: PullImmPortRequest,
    observer: StreamObserver[RawImmPortPayload],
  ): Unit = {
    // TODO: Implement ImmPort API streaming pull
    // 1. Authenticate via ImmPort token
    // 2. Pull dataSets for specified study_ids
    // 3. Map to RawImmPortPayload
    // 4. Stream via observer
    observer.onCompleted()
  }
}
