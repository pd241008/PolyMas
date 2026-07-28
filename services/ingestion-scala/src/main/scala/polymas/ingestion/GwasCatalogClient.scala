package polymas.ingestion

import io.grpc.stub.StreamObserver

class GwasCatalogClient {

  private val baseUrl = sys.env.getOrElse(
    "GWAS_API_BASE_URL",
    "https://www.ebi.ac.uk/gwas/rest/api",
  )

  def streamResults(
    request: PullGwasRequest,
    observer: StreamObserver[RawGwasPayload],
  ): Unit = {
    // TODO: Implement paginated GWAS Catalog REST -> protobuf mapping
    // 1. For each locus_id, hit /singleNucleotidePolymorphisms/{rsId}/associatedRiskFactors
    // 2. Filter by EFO trait IDs and p-value threshold
    // 3. Map JSON response to RawGwasPayload
    // 4. Stream via observer.onNext(), finish with onCompleted()
    observer.onCompleted()
  }
}
