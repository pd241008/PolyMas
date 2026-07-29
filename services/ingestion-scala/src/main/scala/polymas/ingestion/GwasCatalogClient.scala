package polymas.ingestion

import com.google.protobuf.{ByteString, Timestamp}
import com.polymas.proto.v1.*
import com.typesafe.scalalogging.StrictLogging
import io.grpc.stub.StreamObserver
import io.circe.*
import io.circe.generic.auto.*
import io.circe.syntax.*
import sttp.client3.*
import sttp.client3.circe.*

import java.time.Instant
import scala.jdk.CollectionConverters.*

object GwasCatalogClient:
  case class Association(
    rsId: Option[String],
    pvalue: Option[Double],
    pvalueText: Option[String],
    efoTrait: Option[String],
    mappedLabel: Option[String],
    studyId: Option[String],
    orPerCopyNum: Option[Double],
    betaNum: Option[Double],
    standardError: Option[Double],
    range: Option[String],
  )

  case class Embedded(associations: List[Association])
  case class Link(href: String)
  case class Links(self: Option[Link], next: Option[Link])
  case class GwasResponse(_embedded: Embedded, _links: Links)

class GwasCatalogClient extends StrictLogging:
  import GwasCatalogClient.*

  private val baseUrl = sys.env.getOrElse(
    "GWAS_API_BASE_URL",
    "https://www.ebi.ac.uk/gwas/rest/api",
  )

  private val backend = HttpClientSyncBackend()

  def streamResults(
      request: PullGwasRequest,
      observer: StreamObserver[RawGwasPayload],
  ): Unit =
    val locusIds  = request.getLocusIdsList.asScala
    val efoFilter = request.getEfoTraitIdsList.asScala.toSet
    val minPVal   = request.getMinPValue

    if locusIds.isEmpty then
      observer.onCompleted()
      return

    val now = Timestamp.newBuilder()
      .setSeconds(Instant.now.getEpochSecond)
      .setNanos(Instant.now.getNano)
      .build()

    locusIds.foreach { locusId =>
      try
        val filtered = fetchAllPages(locusId)
          .filter(a => passesFilter(a, efoFilter, minPVal))

        if filtered.nonEmpty then
          val jsonBytes = filtered.asJson.noSpaces.getBytes("UTF-8")
          val payload = RawGwasPayload.newBuilder()
            .setSource(s"gwas/$locusId")
            .setRawJson(ByteString.copyFrom(jsonBytes))
            .setFetchedAt(now)
            .build()
          observer.onNext(payload)
      catch case e: Exception =>
        logger.error(s"Failed to fetch GWAS data for locus $locusId", e)
    }

    observer.onCompleted()

  private[ingestion] def passesFilter(
      a: Association,
      efoFilter: Set[String],
      minPVal: Float,
  ): Boolean =
    val efoOk = efoFilter.isEmpty || a.mappedLabel.exists(efoFilter.contains)
    val pOk   = minPVal <= 0f || a.pvalue.exists(_ <= minPVal.toDouble)
    efoOk && pOk

  private def fetchAllPages(locusId: String): List[Association] =
    var result = List.empty[Association]
    var url: Option[String] = Some(
      s"$baseUrl/singleNucleotidePolymorphisms/$locusId/associations",
    )

    while url.isDefined do
      logger.debug("Fetching {}", url.get)
      val response = basicRequest
        .get(uri"${url.get}")
        .header("Accept", "application/json")
        .response(asJson[GwasResponse])
        .send(backend)

      response.body match
        case Right(g) =>
          result ++= g._embedded.associations
          url = g._links.next.map(_.href)
        case Left(e) =>
          logger.error("API error for {}: {}", locusId, e.getMessage)
          url = None

    result