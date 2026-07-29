package polymas.ingestion

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class GwasCatalogClientSpec extends AnyFlatSpec with Matchers:

  "passesFilter" should "allow all when efoFilter is empty and minPVal <= 0" in {
    val c = new GwasCatalogClient
    val a = GwasCatalogClient.Association(None, Some(1e-8), None, None, None, None, None, None, None, None)
    c.passesFilter(a, Set.empty, 0f) should be (true)
  }

  it should "reject above p-value threshold" in {
    val c = new GwasCatalogClient
    val a = GwasCatalogClient.Association(None, Some(1e-3), None, None, None, None, None, None, None, None)
    c.passesFilter(a, Set.empty, 1e-5f) should be (false)
  }

  it should "pass below p-value threshold" in {
    val c = new GwasCatalogClient
    val a = GwasCatalogClient.Association(None, Some(1e-8), None, None, None, None, None, None, None, None)
    c.passesFilter(a, Set.empty, 1e-5f) should be (true)
  }

  it should "ignore p-value filter when minPVal <= 0" in {
    val c = new GwasCatalogClient
    val a = GwasCatalogClient.Association(None, Some(0.5), None, None, None, None, None, None, None, None)
    c.passesFilter(a, Set.empty, 0f) should be (true)
  }

  it should "match when efoTrait is in the filter set" in {
    val c = new GwasCatalogClient
    val a = GwasCatalogClient.Association(None, Some(1e-8), None, None, Some("Blood pressure"), None, None, None, None, None)
    c.passesFilter(a, Set("Blood pressure"), 1e-5f) should be (true)
  }

  it should "reject when efoTrait is not in the filter set" in {
    val c = new GwasCatalogClient
    val a = GwasCatalogClient.Association(None, Some(1e-8), None, None, Some("Diabetes"), None, None, None, None, None)
    c.passesFilter(a, Set("Blood pressure"), 1e-5f) should be (false)
  }

  it should "pass when both efo and p-value filters satisfied" in {
    val c = new GwasCatalogClient
    val a = GwasCatalogClient.Association(None, Some(1e-8), None, None, Some("Blood pressure"), None, None, None, None, None)
    c.passesFilter(a, Set("Blood pressure"), 1e-5f) should be (true)
  }

  it should "reject when efo matches but p-value does not" in {
    val c = new GwasCatalogClient
    val a = GwasCatalogClient.Association(None, Some(1e-3), None, None, Some("Blood pressure"), None, None, None, None, None)
    c.passesFilter(a, Set("Blood pressure"), 1e-5f) should be (false)
  }

  it should "return false for association with no mappedLabel when filter non-empty" in {
    val c = new GwasCatalogClient
    val a = GwasCatalogClient.Association(None, Some(1e-8), None, None, None, None, None, None, None, None)
    c.passesFilter(a, Set("Blood pressure"), 1e-5f) should be (false)
  }