def sample_loop1(svi,Vegvisir,guide,data_loader,args,custom=False):
    Vegvisir.train(False)
    print("Collecting {} samples".format(args.num_samples))
    binary_predictions = []
    logits_predictions = []
    probs_predictions = []
    latent_spaces = []
    with torch.no_grad():
        for batch_number, batch_dataset in enumerate(data_loader):
            batch_data_blosum = batch_dataset["batch_data_blosum"]
            batch_data_int = batch_dataset["batch_data_int"]
            batch_data_onehot = batch_dataset["batch_data_onehot"]
            batch_data_blosum_norm = batch_dataset["batch_data_blosum_norm"]
            batch_mask = batch_dataset["batch_mask"]
            if args.use_cuda:
                batch_data_blosum = batch_data_blosum.cuda()
                batch_data_int = batch_data_int.cuda()
                batch_data_onehot = batch_data_onehot.cuda()
                batch_data_blosum_norm = batch_data_blosum_norm.cuda()
                batch_mask = batch_mask.cuda()
            batch_data = {"blosum": batch_data_blosum, "int": batch_data_int, "onehot": batch_data_onehot,"norm":batch_data_blosum_norm}
            loss = svi.step(batch_data, batch_mask, sample=False)
            sampling_output = Predictive(Vegvisir.model, guide=guide, num_samples=args.num_samples,return_sites=(), parallel=False)(batch_data, batch_mask,sample=True)
            latent_space = sampling_output["latent_z"].squeeze(0).squeeze(0).detach()
            true_labels_batch = batch_data["blosum"][:, 0, 0, 0]
            identifiers = batch_data["blosum"][:, 0, 0, 1]
            partitions = batch_data["blosum"][:, 0, 0, 2]
            immunodominace_score = batch_data["blosum"][:, 0, 0, 4]
            confidence_score = batch_data["blosum"][:, 0, 0, 5]
            latent_space = torch.column_stack(
                [true_labels_batch, identifiers, partitions, immunodominace_score, confidence_score, latent_space])

            binary_class_predictions = sampling_output["predictions"].detach().T
            assert binary_class_predictions.shape == (batch_data["blosum"].shape[0],args.num_samples)
            logits_class_predictions = sampling_output["class_logits"].detach().permute(1,0,2) #N,num_samples,num_classes
            probs_class_predictions = torch.nn.Sigmoid()(logits_class_predictions) #N,num_samples,num_classes #equivalent to torch.exp(logits_class_predictions) / (1 + torch.exp(logits_class_predictions))
            binary_predictions.append(binary_class_predictions)
            logits_predictions.append(logits_class_predictions)
            probs_predictions.append(probs_class_predictions)
            latent_spaces.append(latent_space.detach().cpu().numpy())

    total_binary_predictions = torch.cat(binary_predictions,dim=0)
    total_logits_predictions = torch.cat(logits_predictions,dim=0)
    total_probs_predictions = torch.cat(probs_predictions,dim=0)
    latent_arr = np.concatenate(latent_spaces,axis=0)

    samples_dict = {"binary":total_binary_predictions,
                    "logits":total_logits_predictions,
                    "probs":total_probs_predictions}

    return samples_dict, latent_arr