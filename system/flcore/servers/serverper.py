from flcore.clients.clientper import clientPer
from flcore.servers.serverbase import Server, setup_seed
from threading import Thread
import time


class FedPer(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientPer)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        setup_seed(2022)
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            # for the 0-th round all clients use the same head;
            # after the self.aggregate_parameters() (defined in this file.) the global model only attains a base part.
            # hence no head is being set to selected clients after 0-th round
            self.send_models()
            # print('-1', self.global_model.state_dict()['predictor.weight'][0, :3])

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                cids = []
                for c in self.selected_clients:
                    cids.append(c.id)
                print("Selected clients", cids)
                # print("\nEvaluate global model")
                # this is incorrect in the sense that the global model is tested on a subset of data.
                # self.evaluate()

            for client in self.selected_clients:
                # print(client.id, client.model.state_dict()['predictor.weight'][0, :3])
                client.train()

            if i % self.eval_gap == 0:
                print("\nEvaluate local models")
                self.evaluate()

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))

        self.save_results()
        self.save_global_model()

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_weights = []
        tot_samples = 0
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples)
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            # only body is uploaded
            self.uploaded_models.append(client.model.base)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
