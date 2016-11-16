package de.akquinet.ats.ak40.weka;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

import javax.swing.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Random;

public class WekaJ48Demo {


    public void testWekaJ48() throws Exception {
        Instances data = getData("/iris.arff", 1);
        // mit numInstances() kann die Zahl der Zeilen im Datensatz ausgegeben werden
        System.out.println(data.numInstances() + " Zeilen im Datensatz");
        // ZeroR: Simpelster Klassifikator
        ZeroR simplestClassifier = new ZeroR();
        // J48: Entscheidungsbaum-Algorithmus
        J48 decisionTree = new J48();

        String[] options = new String[2];
        // Minimale Fälle pro Blatt: -M
        options[0] = "-M";
        // Anzahl 5
        options[1] = "5";
        decisionTree.setOptions(options);

        // build classifier
        decisionTree.buildClassifier(data);
        // print classifier
        System.out.println(decisionTree);

        // Anzahl der Iterationen für die Kreuzvalidierung
        Integer numIterations = 10;
        Random randData = new Random(1);
        Evaluation evalSimpClass = evalModel(simplestClassifier, data, numIterations, randData);
        Evaluation evalTree = evalModel(decisionTree, data, numIterations, randData);
        // Modellfit Simpler Klassifikator
        System.out.println("Modellfit Simpler Klassifikator: \n" + evalSimpClass.toSummaryString());
        // Konfusionsmatrix
        System.out.println(evalSimpClass.toMatrixString());
        // Modellfit Entscheidungsbaum
        System.out.println("Modellfit Entscheidungsbaum: \n" + evalTree.toSummaryString());
        // Konfusionsmatrix
        System.out.println(evalTree.toMatrixString());

        // display classifier
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                TreeVisualizer visualizeTree = null;
                try {
                    visualizeTree = new TreeVisualizer( null, decisionTree.graph(), new PlaceNode2());
                    final JFrame jFrame = new JFrame("Weka J48 Klassisfikator: Entscheidungsbaum");
                    jFrame.setSize(600,500);
                    jFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                    jFrame.getContentPane().add(visualizeTree);
                    jFrame.setVisible(true);
                    visualizeTree.fitToScreen();
                } catch ( Exception e ) {
                    e.printStackTrace();
                }
            }
        });

        Instances newData = getData("/iris_new.arff", 1);
        for (int i = 0; i < newData.numInstances(); i++) {
            double result = decisionTree.classifyInstance(newData.instance(i));
            newData.instance(i).setValue(newData.numAttributes() - 1, newData.classAttribute().value((int) result));
        }
        System.out.println(newData);

    }

    /** Erzeugt ein Evaluation Objekt, mit dem der Klassifikator auf die gegebenen Daten angewendet wird.
     *
     * @param classifier Der Klassifikator
     * @param data Die Daten
     * @param numberIterations Die Anzahl der Unterteilungen des Datensatzes während der Kreuzvalidierung
     * @param randData Ein Zufallszahlengenerator
     * @return Das Evaluation Objekt
     */
    private Evaluation evalModel(
            Classifier classifier, Instances data, Integer numberIterations, Random randData ) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(classifier, data, numberIterations, randData);
        return eval;
    }

    /** Liest aus einer ARFF Datei Daten mit Attribut- und Datenbeschreibungen
     *
     * @param filename Pfad und Name der Datei
     * @param posClass 1-basierter Index der Klassendefinition vom Ende der Attributliste aus gesehen.
     * @return Ein Instances Objekt
     */
    private Instances getData( String filename, Integer posClass ) throws IOException, URISyntaxException {
        // Einlesen der Daten
        File file = new File(WekaJ48Demo.class.getResource( filename ).toURI());
        BufferedReader inputReader = new BufferedReader(new FileReader(file));
        // Erstelle einen Datensatz der Klasse Instances
        Instances data = new Instances(inputReader);
        // Bestimme letztes Attribut als Zielklasse
        data.setClassIndex(data.numAttributes() - posClass);

        return data;
    }


}
