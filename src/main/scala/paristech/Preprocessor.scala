package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{udf, datediff, second, round, concat_ws, lower, from_unixtime}

object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()

    /** *****************************************************************************
      *
      * TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      * if problems with unimported modules => sbt plugins update
      *
      * *******************************************************************************/
    import spark.implicits._

    println("\n")
    println("Hello World ! from Preprocessor")
    println("\n")

    val df: DataFrame = spark.read.option("header", true).option("inferSchema", true).csv("data/train_clean.csv")
    println(s"Nombre de linges : ${df.count}")
    println(s"Nombre de colonnes : ${df.columns.length}")

    df.show()
    df.printSchema()

    val dfCasted: DataFrame = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline", $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))

    dfCasted.printSchema()

    dfCasted
      .select("goal", "backers_count", "final_status")
      .describe()
      .show()

    dfCasted
      .groupBy("country")
      .count()
      .orderBy($"count".desc)
      .show(10)

    dfCasted
      .select("deadline")
      .dropDuplicates()
      .show()

    val df2: DataFrame = dfCasted.drop("disable_communication")
    df2.printSchema()

    val dfNoFuture: DataFrame = df2.drop("backers_count", "state_changed_at")

    dfNoFuture
      .filter($"country" === "False") //attention "===" (= * 3)
      .groupBy($"currency")
      .count()
      .orderBy($"count".desc)
      .show(50)

    val cleanCountryUdf = udf(cleanCountry _)
    val cleanCurrencyUdf = udf(cleanCurrency _)
    val dfCountry: DataFrame = dfNoFuture
      .withColumn("country2", cleanCountryUdf($"country", $"currency"))
      .withColumn("currency2", cleanCurrencyUdf($"currency"))
      .drop("country", "currency")

    dfCountry.show()

    val dfFinal: DataFrame = dfCountry
      .withColumn("days_campaign", datediff(from_unixtime($"deadline"), from_unixtime($"launched_at")))
      .withColumn("hours_prepa", round(second(from_unixtime($"launched_at" - $"created_at")) / 3600, 3))
      .drop("created_at", "deadline", "launched_at")
      .withColumn("text", lower(concat_ws(" ", $"name", $"desc", $"keywords")))
      .na.fill(Map("days_campaign" -> -1, "hours_prepa" -> -1, "goal" -> -1, "country2" -> "unknown", "currency2" -> "unknown"))
    dfFinal.show()

    dfFinal.write.parquet("data/tp2_df_final.parquet")
  }

  def cleanCountry(country: String, currency: String): String = {
    if (country == "False")
      currency
    else if (country != null && country.length != 2)
      null
    else
      country
  }

  def cleanCurrency(currency: String): String = {
    if (currency != null && currency.length != 3)
      null
    else
      currency
  }

}
