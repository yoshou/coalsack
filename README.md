# coalsack

Distributed graph processing library

## Install 
```console
$ git clone https://github.com/yoshou/coalsack.git
$ cd coalsack && mkdir build && cd build
$ cmake .. && make -j
```

## Dependencies:
* [boost](https://www.boost.org)
* [cereal](https://github.com/USCiLab/cereal.git)
* [spdlog](https://github.com/gabime/spdlog.git)
 
## Usage samples

#### Basic usage
```c++

int main(int argc, char *argv[])
{
    std::shared_ptr<subgraph> g1(new subgraph());
    std::shared_ptr<subgraph> g2(new subgraph());

    std::shared_ptr<text_heartbeat_node> n1(new text_heartbeat_node());
    n1->set_message("Hello!!\n");
    g1->add_node(n1);

    std::shared_ptr<p2p_talker_node> n2(new p2p_talker_node());
    n2->set_input(n1->get_output());
    g1->add_node(n2);

    std::shared_ptr<p2p_listener_node> n3(new p2p_listener_node());
    n3->set_input(n2->get_output());
    n3->set_endpoint("127.0.0.1", 0);
    g2->add_node(n3);

    std::shared_ptr<passthrough_node> n4(new passthrough_node());
    n4->set_input(n3->get_output());
    g2->add_node(n4);

    std::shared_ptr<buffer_node> n5(new buffer_node());
    n5->set_input(n4->get_output());
    n5->set_interval(2000);
    g2->add_node(n5);

    std::shared_ptr<console_node> n6(new console_node());
    n6->set_input(n5->get_output());
    g2->add_node(n6);

    asio::io_service io_service;
    graph_proc_client client;
    client.deploy(io_service, "127.0.0.1", 31400, g1);
    client.deploy(io_service, "127.0.0.1", 31400, g2);

    client.run();
    io_service.run();

    return 0;
}

```