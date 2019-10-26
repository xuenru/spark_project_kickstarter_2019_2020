package paristech

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, IDF, OneHotEncoderEstimator, RegexTokenizer, StopWordsRemover, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()


    /** *****************************************************************************
      *
      * TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      * if problems with unimported modules => sbt plugins update
      *
      * *******************************************************************************/

    println("hello world ! from Trainer")

    // Charger le DataFrame obtenu à la fin du TP 2.
    val data = spark.read.parquet("data/tp2_df_final.parquet")
    data.show()

    // Stage 1 : récupérer les mots des textes
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")
    val dataTokens = tokenizer.transform(data)
    dataTokens.select("tokens").show()

    // Stage 2 : retirer les stop words
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("words")

    val dataWords = stopWordsRemover.transform(dataTokens)
    dataWords.select("words").show(false)

    // Stage 3 : computer la partie TF
    val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("TF")
      .setVocabSize(3)
      .setMinDF(2)
      .fit(dataWords)

    val dataTF = cvModel.transform(dataWords)
    dataTF.select("TF").show(false)

    val idf = new IDF()
      .setInputCol("TF")
      .setOutputCol("tfidf")

    val idfModel = idf.fit(dataTF)

    val dataTFIDF = idfModel.transform(dataTF)
    dataTFIDF.select("tfidf").show(false)

    val countryIndexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    val DataCountryIndexed = countryIndexer.fit(dataTFIDF).transform(dataTFIDF)
    DataCountryIndexed.show()

    val currencyIndexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    val DataCCIndexed = currencyIndexer.fit(DataCountryIndexed).transform(DataCountryIndexed)
    DataCCIndexed.select("country2", "currency2", "country_indexed", "currency_indexed").show()

    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed", "currency_indexed"))
      .setOutputCols(Array("country_onehot", "currency_onehot"))
    val dataOneHot = encoder.fit(DataCCIndexed).transform(DataCCIndexed)
    dataOneHot.select("country_indexed", "currency_indexed", "country_onehot", "currency_onehot").show()


    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

    val dataAssembled = assembler.transform(dataOneHot)
    dataAssembled.select("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot", "features").show()
    dataAssembled.select("features").show(false)

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_prediction")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(20)
  }
}
