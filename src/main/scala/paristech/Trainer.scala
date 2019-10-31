package paristech

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, IDF, OneHotEncoderEstimator, RegexTokenizer, StopWordsRemover, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}


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

    import spark.implicits._

    println("hello world ! from Trainer")

    // Charger le DataFrame obtenu à la fin du TP 2.
    val dataRaw = spark.read.parquet("data/tp2_df_final.parquet")
    //    dataRaw.show()
    // clean the bad lines
    val data: DataFrame = dataRaw
      .filter($"days_campaign" =!= -1)
      .filter($"hours_prepa" =!= -1)
      .filter($"goal" =!= -1)
      .filter($"country2" =!= "unknown")
      .filter($"currency2" =!= "unknown")

    // Stage 1 : récupérer les mots des textes
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")
    val dataTokens = tokenizer.transform(data)
    //    dataTokens.select("tokens").show()

    // Stage 2 : retirer les stop words
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("words")

    val dataWords = stopWordsRemover.transform(dataTokens)
    //    dataWords.select("words").show(false)

    // Stage 3 : computer la partie TF
    val cv = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("TF")
      .setVocabSize(20)
      .setMinDF(2)
    val cvModel: CountVectorizerModel = cv.fit(dataWords)


    val dataTF = cvModel.transform(dataWords)
    //    dataTF.select("TF").show(false)

    val idf = new IDF()
      .setInputCol("TF")
      .setOutputCol("tfidf")

    val idfModel = idf.fit(dataTF)

    val dataTFIDF = idfModel.transform(dataTF)
    //    dataTFIDF.select("tfidf").show(false)

    val countryIndexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    val DataCountryIndexed = countryIndexer.fit(dataTFIDF).transform(dataTFIDF)
    //    DataCountryIndexed.show()

    val currencyIndexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    val DataCCIndexed = currencyIndexer.fit(DataCountryIndexed).transform(DataCountryIndexed)
    //    DataCCIndexed.select("country2", "currency2", "country_indexed", "currency_indexed").show()

    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed", "currency_indexed"))
      .setOutputCols(Array("country_onehot", "currency_onehot"))
    val dataOneHot = encoder.fit(DataCCIndexed).transform(DataCCIndexed)
    //    dataOneHot.select("country_indexed", "currency_indexed", "country_onehot", "currency_onehot").show()


    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

    val dataAssembled = assembler.transform(dataOneHot)
    //    dataAssembled.select("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot", "features").show()
    //    dataAssembled.select("features").show(false)
    //    dataAssembled.select("features", "final_status").show(false)

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0) // 设置ElasticNet混合参数,范围为[0，1]。// 对于α= 0，惩罚是L2惩罚。 对于alpha = 1，它是一个L1惩罚。 对于0 <α<1，惩罚是L1和L2的组合。 默认值为0.0，这是一个L2惩罚。
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true) // 在拟合模型之前,是否标准化特征
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_prediction")
      .setThresholds(Array(0.7, 0.3)) // 在二进制分类中设置阈值，范围为[0，1]。如果类标签1的估计概率>Threshold，则预测1，否则0.高阈值鼓励模型更频繁地预测0; 低阈值鼓励模型更频繁地预测1。默认值为0.5。
      .setTol(1.0e-6) // 设置迭代的收敛容限。 较小的值将导致更高的精度与更多的迭代的成本。 默认值为1E-6。
      .setMaxIter(20)

    // Pipeline
    val pileline = new Pipeline().setStages(
      Array(tokenizer, stopWordsRemover, cv, idf, countryIndexer, currencyIndexer, encoder, assembler, lr))

    val Array(train, test) = data.randomSplit(Array(0.9, 0.1))

    val pipelineModel = pileline.fit(train)
    val dfWithSimplePredictions = pipelineModel.transform(test)

    val f1score = new MulticlassClassificationEvaluator()
      .setMetricName("f1")
      .setPredictionCol("predictions")
      .setLabelCol("final_status")
    pipelineModel.transform(test).select("final_status", "features", "raw_prediction", "probability", "predictions").show()
    println(f1score.evaluate(dfWithSimplePredictions))
  }
}
