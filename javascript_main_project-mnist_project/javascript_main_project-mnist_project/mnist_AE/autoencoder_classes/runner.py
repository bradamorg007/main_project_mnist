import autoencoder_classes.convolutional as CNN_network
import autoencoder_classes.fully_connected as FE_Network
import autoencoder_classes.cluster_analysis as ClusterAnalysis

if __name__ == '__main__':

    CNN_network.run_main()
    FE_Network.run_main()
    ClusterAnalysis.run_main()

    print()
    print('===============================================================================================')
    print('FULL RUNNER COMPLETE')
    print('===============================================================================================')
    print()