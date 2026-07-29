package polymas.ingestion

import com.google.protobuf.{ByteString, Timestamp}
import com.polymas.proto.v1.*
import com.typesafe.scalalogging.StrictLogging
import io.grpc.stub.StreamObserver
import io.circe.*
import io.circe.parser.*
import io.circe.syntax.*
import sttp.client3.*

import java.time.Instant
import scala.jdk.CollectionConverters.*

class ImmPortClient(
    baseUrl: String = sys.env.getOrElse(
      "IMMPORT_API_BASE_URL",
      "https://www.immport.org/data/query",
    ),
    apiKey: String = sys.env.getOrElse("IMMPORT_API_KEY", ""),
) extends StrictLogging:

  private val backend = HttpClientSyncBackend()

  def streamResults(
      request: PullImmPortRequest,
      observer: StreamObserver[RawImmPortPayload],
  ): Unit =
    val studyIds   = request.getStudyIdsList.asScala
    val assayTypes = request.getAssayTypesList.asScala.toSet

    if studyIds.isEmpty then
      observer.onCompleted()
      return

    val now = Timestamp.newBuilder()
      .setSeconds(Instant.now.getEpochSecond)
      .setNanos(Instant.now.getNano)
      .build()

    studyIds.foreach { studyId =>
      try
        fetchJson(uri"$baseUrl/api/study/$studyId?format=json") match
          case Some(studyJson) =>
            val resultJson =
              if assayTypes.nonEmpty then mergeAssessments(studyJson, studyId)
              else studyJson

            observer.onNext(
              RawImmPortPayload.newBuilder()
                .setSource(s"immport/$studyId")
                .setRawJson(ByteString.copyFromUtf8(resultJson))
                .setFetchedAt(now)
                .build(),
            )
          case None =>
            logger.warn("No data returned for ImmPort study {}", studyId)
      catch case e: Exception =>
        logger.error(s"Failed to fetch ImmPort data for study $studyId", e)
    }

    observer.onCompleted()

  private[ingestion] def authHeaders: Map[String, String] =
    if apiKey.nonEmpty then Map("Authorization" -> s"Bearer $apiKey")
    else
      logger.warn("IMMPORT_API_KEY not set — requests may be rejected")
      Map.empty

  private def fetchJson(uri: sttp.model.Uri): Option[String] =
    val response = basicRequest
      .get(uri)
      .headers(authHeaders)
      .response(asString)
      .send(backend)

    response.body match
      case Right(body) =>
        if response.code.isSuccess then Some(body)
        else
          logger.error("ImmPort API {} {}: {}", response.code.code, uri, body.take(200))
          None
      case Left(e) =>
        logger.error("ImmPort API request failed for {}: {}", uri, e)
        None

  private def mergeAssessments(studyJson: String, studyId: String): String =
    fetchJson(uri"$baseUrl/api/study/assessment/$studyId?format=json") match
      case Some(assessmentJson) =>
        val merged = for
          studyObj      <- parse(studyJson).flatMap(_.as[JsonObject])
          assessmentObj <- parse(assessmentJson).flatMap(_.as[JsonObject])
        yield studyObj.add("assessments", assessmentObj.asJson).asJson.noSpaces

        merged match
          case Right(json) => json
          case Left(e) =>
            logger.error("Failed to merge assessment JSON for {}: {}", studyId, e.getMessage)
            studyJson
      case None => studyJson
