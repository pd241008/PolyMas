package polymas.ingestion

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ImmPortClientSpec extends AnyFlatSpec with Matchers:

  "authHeaders" should "return Bearer header when apiKey is set" in {
    val client = new ImmPortClient(baseUrl = "https://www.immport.org/data/query", apiKey = "test-key-123")
    client.authHeaders should be (Map("Authorization" -> "Bearer test-key-123"))
  }

  it should "return empty map when apiKey is empty" in {
    val client = new ImmPortClient(baseUrl = "https://www.immport.org/data/query", apiKey = "")
    client.authHeaders should be (Map.empty)
  }