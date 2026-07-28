lazy val root = (project in file("."))
  .settings(
    name         := "polymas-ingestion",
    version      := "0.1.0-SNAPSHOT",
    scalaVersion := "3.4.2",
    Compile / unmanagedSourceDirectories += baseDirectory.value / "../../proto/gen/java",
    libraryDependencies ++= Seq(
      "io.grpc"                  % "grpc-netty"   % "1.64.0",
      "io.grpc"                  % "grpc-protobuf" % "1.64.0",
      "io.grpc"                  % "grpc-stub"    % "1.64.0",
      "io.grpc"                  % "grpc-api"     % "1.64.0",
      "com.google.protobuf"      % "protobuf-java" % "3.25.3",
      "com.softwaremill.sttp.client3" %% "core"   % "3.9.7",
      "com.softwaremill.sttp.client3" %% "circe"  % "3.9.7",
      "io.circe"                 %% "circe-core"   % "0.14.7",
      "io.circe"                 %% "circe-generic" % "0.14.7",
      "io.circe"                 %% "circe-parser" % "0.14.7",
      "com.typesafe.scala-logging" %% "scala-logging" % "3.9.5",
      "ch.qos.logback"            % "logback-classic" % "1.5.6",
      "org.scalatest"            %% "scalatest"   % "3.2.18" % Test,
    ),
  )
