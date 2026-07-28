FROM eclipse-temurin:21-jdk AS build

WORKDIR /app
COPY build.sbt project/ ./
RUN sbt update

COPY src/ src/
RUN sbt compile

FROM eclipse-temurin:21-jre
WORKDIR /app
COPY --from=build /app/target/scala-3.4/polymas-ingestion-assembly-*.jar /app/server.jar
EXPOSE 50051
ENTRYPOINT ["java", "-jar", "/app/server.jar"]
