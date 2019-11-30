package paristech

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, IDF, OneHotEncoderEstimator, RegexTokenizer, StopWordsRemover, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
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
    val dataRaw = spark.read.parquet("src/main/resources/preprocessed.parquet")
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
    // dataTokens.select("text", "tokens").show()

    // Stage 2 : retirer les stop words
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("words")

    val dataWords = stopWordsRemover.transform(dataTokens)
    // dataWords.select("words").show(false)

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

    // A label indexer that maps a string column of labels to an ML column of label indices.
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
    //        dataOneHot.select("country_indexed", "currency_indexed", "country_onehot", "currency_onehot").show()

    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

    val dataAssembled = assembler.transform(dataOneHot)
    //    dataAssembled.select("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot", "features").show()
    //    dataAssembled.select("features").show(false)
    //    dataAssembled.select("features", "final_status").show(false)

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0) // set ElasticNet, value [0, 1], 0=> L2(default), 1=> L1
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true) // before fit if do Standardization
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_prediction")
      .setThresholds(Array(0.7, 0.3)) // default 0.5, [0，1]. if value > Threshold then result = 1
      //      .setThreshold(0.3) // some as function above
      .setTol(1.0e-6) // like epsilon default 1E-6
      .setMaxIter(20)

    // Pipeline
    val pipeline = new Pipeline().setStages(
      Array(tokenizer, stopWordsRemover, cv, idf, countryIndexer, currencyIndexer, encoder, assembler, lr))

    val Array(train, test) = data.randomSplit(Array(0.9, 0.1))

    val pipelineModel = pipeline.fit(train)
    val dfWithSimplePredictions = pipelineModel.transform(test)

    // f1score = 2 * (precision*recall) / (precision+recall)
    val f1score = new MulticlassClassificationEvaluator()
      .setMetricName("f1")
      .setPredictionCol("predictions")
      .setLabelCol("final_status")
    pipelineModel.transform(test).select("final_status", "features", "raw_prediction", "probability", "predictions").show()
    println("f1score： " + f1score.evaluate(dfWithSimplePredictions))

    // Réglage des hyper-paramètres (a.k.a. tuning) du modèle
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(cv.minDF, Array(55.0, 75.0, 95.0))
      .addGrid(cv.vocabSize, Array(3, 10, 20, 30, 40))
      .build()

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)
      .setEvaluator(f1score)
    val model = trainValidationSplit.fit(train)

    val dfWithPredictions = model.transform(test)

    dfWithPredictions.select("final_status", "features", "raw_prediction", "probability", "predictions").show()
    dfWithPredictions.groupBy("final_status", "predictions").count.show()
    // print f1score
    println("f1score： " + model.getEvaluator.evaluate(dfWithPredictions))
    model.save("src/main/resources/trained_model")
    //for loading model
    //    val xModel = TrainValidationSplitModel.load("data/tp3_model")

  }
}
