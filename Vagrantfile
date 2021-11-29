Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/focal64"
  config.vm.provision :shell, path: "Vagrant/bootstrap.sh"
  config.vm.provision :shell, privileged: false, path: "Vagrant/user.sh"
  config.vm.network "forwarded_port", guest: 5002, host: 5010, auto_correct: true, id: "rasa X"
  config.vm.network "forwarded_port", guest: 8000, host: 8010, auto_correct: true, id: "Facebook Duckling"

  config.vm.provider "virtualbox" do |v|
        v.name = "chatbot-test" # friendly name that shows up in Oracle VM VirtualBox Manager
        v.memory = 3072
        v.cpus = 2
        v.gui = true
        # v.customize ["modifyvm", :id, "--natdnshostresolver1", "on"] # fixes slow dns lookups
    end
end
