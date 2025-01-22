#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <vector>
#include <iostream>
#include <map>
#include <stdexcept>
#include <functional>
#include <thread>
#include <atomic>
#include <sstream>

#include <cereal/cereal.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/archives/binary.hpp>

#include <spdlog/spdlog.h>

#include "utils.h"
#include "rpc_server.h"
#include "rpc_client.h"
#include "data_stream_transmitter.h"
#include "data_stream_receiver.h"

namespace coalsack
{
    class graph_message
    {
    public:
        graph_message() = default;
        virtual ~graph_message() = default;

        template <typename Archive>
        void serialize(Archive &archive)
        {
        }
    };

    using graph_message_ptr = std::shared_ptr<graph_message>;

    struct graph_message_callback
    {
        using func_type = std::function<void(graph_message_ptr)>;
        func_type func;

        graph_message_callback(func_type func)
            : func(func)
        {
        }

        virtual void operator()(graph_message_ptr message)
        {
            func(message);
        }

        virtual ~graph_message_callback() = default;
    };

    class graph_node;

    enum class EDGE_TYPE
    {
        DATAFLOW = 0,
        CHAIN = 1,
    };

    class subscribe_request
    {
        std::string proc_name;
        std::string msg_type;
        std::vector<uint8_t> data;

    public:
        subscribe_request()
        {
        }

        void set_proc_name(std::string value)
        {
            proc_name = value;
        }
        std::string get_proc_name() const
        {
            return proc_name;
        }

        void set_msg_type(std::string value)
        {
            msg_type = value;
        }
        std::string get_msg_type() const
        {
            return msg_type;
        }

        void set_data(const std::vector<uint8_t> &value)
        {
            data = value;
        }
        std::vector<uint8_t> get_data() const
        {
            return data;
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(proc_name, msg_type, data);
        }
    };

    class graph_edge
    {
        graph_node *source;
        std::string name;
        EDGE_TYPE edge_type;
        std::vector<std::shared_ptr<graph_message_callback>> callbacks;

    public:
        graph_edge(graph_node *source, EDGE_TYPE edge_type = EDGE_TYPE::DATAFLOW)
            : source(source), name(), edge_type(edge_type), callbacks()
        {
        }

        void set_name(std::string name)
        {
            this->name = name;
        }

        std::string get_name() const
        {
            return name;
        }

        EDGE_TYPE get_type() const
        {
            return edge_type;
        }

        graph_node *get_source() const
        {
            return source;
        }

        void set_callback(std::shared_ptr<graph_message_callback> callback)
        {
            if (std::find(callbacks.begin(), callbacks.end(), callback) != callbacks.end())
            {
                throw std::logic_error("The callback has been already registerd");
            }
            callbacks.push_back(callback);
        }

        void remove_callback()
        {
            callbacks.clear();
        }

        void send(graph_message_ptr message)
        {
            for (const auto &callback : callbacks)
            {
                if (callback)
                {
                    (*callback)(message);
                }
            }
        }

        subscribe_request request;
    };

    using graph_edge_ptr = std::shared_ptr<graph_edge>;

    class subgraph;

    class resource_base
    {
    public:
        virtual std::string get_name() const = 0;
        virtual ~resource_base() = default;
    };

    class resource_list
    {
        std::unordered_map<std::string, std::shared_ptr<resource_base>> resources;

    public:
        void add(const std::shared_ptr<resource_base> &resource)
        {
            resources.insert(std::make_pair(resource->get_name(), resource));
        }

        std::shared_ptr<resource_base> get(const std::string &name) const
        {
            const auto found = resources.find(name);
            if (found == resources.end())
            {
                return nullptr;
            }
            return found->second;
        }
    };

    class graph_node
    {
    protected:
        subgraph *g;
        std::shared_ptr<resource_list> resources;

        void set_output(graph_edge_ptr output)
        {
            set_output(output, "default");
        }

        void set_output(graph_edge_ptr output, std::string name)
        {
            output->set_name(name);
            outputs.insert(std::make_pair(name, output));
        }

    private:
        std::unordered_map<std::string, graph_edge_ptr> outputs;
        std::unordered_map<std::string, graph_edge_ptr> inputs;

    public:
        virtual std::string get_proc_name() const = 0;

        graph_node()
            : g(nullptr), outputs(), inputs()
        {
        }

        virtual ~graph_node()
        {
            std::string input_name;
            graph_edge_ptr input_edge;
            for (auto input : inputs)
            {
                std::tie(input_name, input_edge) = input;

                input_edge->remove_callback();
            }
        }

        void set_resources(const std::shared_ptr<resource_list> &resources)
        {
            this->resources = resources;
        }
        void set_input(graph_edge_ptr input)
        {
            set_input(input, "default");
        }

        void set_input(graph_edge_ptr input, std::string name)
        {
            inputs.insert(std::make_pair(name, input));

            input->set_callback(std::make_shared<graph_message_callback>([this, name](graph_message_ptr msg)
                                                                         { process(name, msg); }));
        }

        const std::unordered_map<std::string, graph_edge_ptr> &get_outputs() const
        {
            return outputs;
        }

        graph_edge_ptr get_output() const
        {
            return get_output("default");
        }

        graph_edge_ptr get_output(std::string name) const
        {
            if (outputs.find(name) == outputs.end())
            {
                throw std::invalid_argument("name");
            }
            return outputs.at(name);
        }

        graph_edge_ptr get_input() const
        {
            return get_input("default");
        }
        graph_edge_ptr get_input(std::string name) const
        {
            if (inputs.find(name) == inputs.end())
            {
                throw std::invalid_argument("name");
            }
            return inputs.at(name);
        }
        const std::unordered_map<std::string, graph_edge_ptr> &get_inputs() const
        {
            return inputs;
        }

        virtual void process(std::string input_name, graph_message_ptr message)
        {
        }

        void set_parent(subgraph *g)
        {
            this->g = g;
        }

        subgraph *get_parent() const
        {
            return this->g;
        }

        virtual void initialize()
        {
        }

        virtual void finalize()
        {
        }

        virtual void run()
        {
        }

        virtual void stop()
        {
        }
    };

    using graph_node_ptr = std::shared_ptr<graph_node>;
        
    template <typename T>
    static void dfs_postorder(graph_node *node, std::unordered_set<graph_node *> &visited, T callback)
    {
        if (node == nullptr)
        {
            return;
        }
        if (visited.find(node) != visited.end())
        {
            return;
        }

        visited.insert(node);

        for (auto input : node->get_inputs())
        {
            const auto &[input_name, input_edge] = input;
            dfs_postorder(input_edge->get_source(), visited, callback);
        }

        callback(node);
    }

    class subgraph
    {
        std::vector<graph_node_ptr> nodes;
        std::unordered_map<const graph_node *, uint32_t> node_ids;

    public:
        subgraph()
        {
        }

        void add_node(graph_node_ptr node)
        {
            if (node->get_parent())
            {
                throw std::invalid_argument("The node already belongs to another graph.");
            }
            node_ids.insert(std::make_pair(node.get(), (uint32_t)nodes.size() + 1));
            nodes.push_back(node);
            node->set_parent(this);
        }

        uint32_t get_node_id(const graph_node *node) const
        {
            return node_ids.at(node);
        }

        uint32_t get_node_count() const
        {
            return nodes.size();
        }

        graph_node_ptr get_node(uint32_t idx) const
        {
            return nodes[idx];
        }

        template <typename Archive>
        void save(Archive &archive) const
        {
            archive(nodes);

            using node_port = std::tuple<uint32_t, std::string>;

            std::vector<std::tuple<node_port, node_port, EDGE_TYPE>> edges;
            for (size_t i = 0; i < nodes.size(); i++)
            {
                auto target = nodes[i];

                std::string target_input_name;
                graph_edge_ptr target_input;
                for (auto input : target->get_inputs())
                {
                    std::tie(target_input_name, target_input) = input;

                    if (target_input->get_type() == EDGE_TYPE::DATAFLOW)
                    {
                        node_port target_port((uint32_t)i, target_input_name);

                        auto source = target_input->get_source();
                        auto source_output_name = target_input->get_name();

                        auto j = std::distance(nodes.begin(), std::find_if(nodes.begin(), nodes.end(), [source](graph_node_ptr node)
                                                                           { return node.get() == source; }));

                        assert(nodes[j].get() == source);

                        node_port source_port((uint32_t)j, source_output_name);

                        edges.push_back(std::make_tuple(source_port, target_port, EDGE_TYPE::DATAFLOW));
                    }
                    else if (target_input->get_type() == EDGE_TYPE::CHAIN)
                    {
                        node_port target_port((uint32_t)i, target_input_name);

                        auto source_output_name = target_input->get_name();

                        node_port source_port((uint32_t)-1, source_output_name);

                        edges.push_back(std::make_tuple(source_port, target_port, EDGE_TYPE::CHAIN));
                    }
                }
            }

            archive(edges);
        }

        template <class Archive>
        void load(Archive &archive)
        {
            archive(nodes);

            for (size_t i = 0; i < nodes.size(); i++)
            {
                auto node = nodes[i];
                node_ids.insert(std::make_pair(node.get(), (uint32_t)i + 1));
                node->set_parent(this);
            }

            using node_port = std::tuple<uint32_t, std::string>;

            std::vector<std::tuple<node_port, node_port, EDGE_TYPE>> edges;
            archive(edges);

            for (auto edge : edges)
            {
                node_port source_port, target_port;
                EDGE_TYPE edge_type;
                std::tie(source_port, target_port, edge_type) = edge;

                if (edge_type == EDGE_TYPE::DATAFLOW)
                {
                    auto j = std::get<0>(source_port);
                    auto i = std::get<0>(target_port);

                    auto source = nodes[j];
                    auto target = nodes[i];

                    auto target_input_name = std::get<1>(target_port);
                    auto source_output_name = std::get<1>(source_port);

                    target->set_input(source->get_output(source_output_name), target_input_name);
                }
                else if (edge_type == EDGE_TYPE::CHAIN)
                {
                    auto j = std::get<0>(source_port);
                    auto i = std::get<0>(target_port);

                    assert(j == (uint32_t)-1);

                    auto target = nodes[i];

                    auto target_input_name = std::get<1>(target_port);
                    auto source_output_name = std::get<1>(source_port);

                    auto external_output = std::make_shared<graph_edge>(nullptr);
                    external_output->set_name(source_output_name);

                    target->set_input(external_output, target_input_name);
                }
            }
        }

        void save_to(std::ostream &s) const
        {
            cereal::BinaryOutputArchive oarchive(s);
            oarchive(*this);
        }

        void load_from(std::istream &s)
        {
            cereal::BinaryInputArchive iarchive(s);
            iarchive(*this);
        }

        void run()
        {
            std::unordered_set<graph_node *> visited;
            std::vector<graph_node *> ordered_nodes;
            for (auto& node : nodes)
            {
                dfs_postorder(node.get(), visited, [&ordered_nodes](graph_node *node)
                              { ordered_nodes.push_back(node); });
            }
            std::reverse(ordered_nodes.begin(), ordered_nodes.end());
            for (auto node : ordered_nodes)
            {
                node->run();
            }
        }

        void stop()
        {
            for (auto node : nodes)
            {
                node->stop();
            }
        }

        void validate()
        {
            for (size_t i = 0; i < nodes.size(); i++)
            {
                auto target = nodes[i];

                std::string target_input_name;
                graph_edge_ptr target_input;
                for (auto input : target->get_inputs())
                {
                    std::tie(target_input_name, target_input) = input;

                    if (target_input->get_type() == EDGE_TYPE::DATAFLOW)
                    {
                        auto source = target_input->get_source();
                        if (source->get_parent() != this)
                        {
                            throw std::logic_error("Invalid input");
                        }
                    }
                }
            }
        }
    };

    enum class GRAPH_PROC_RPC_FUNC : uint32_t
    {
        DEPLOY = 0,
        INITIALIZE = 1,
        RUN = 2,
        STOP = 3,
        FINALIZE = 4,
        PROCESS = 5,
    };

    class graph_proc
    {
        std::shared_ptr<subgraph> g;
        std::shared_ptr<resource_list> resources;

    public:
        graph_proc(const std::shared_ptr<resource_list> &resources = std::make_shared<resource_list>())
            : g(new subgraph()), resources(resources)
        {
        }

        std::shared_ptr<subgraph> get_graph() const
        {
            return g;
        }

        std::shared_ptr<resource_list> get_resources() const
        {
            return resources;
        }
        
        void deploy(const std::shared_ptr<subgraph> &g)
        {
            this->g = g;
        }

        void run()
        {
            initialize();
            run(g.get());
        }

        void stop()
        {
            stop(g.get());
            finalize();
        }

        void process(const graph_node *node, const graph_message_ptr &message)
        {
            process(node, "default", message);
        }

        void process(const graph_node *node, const std::string &input_name, const graph_message_ptr &message)
        {
            auto g = node->get_parent();
            auto node_id = node->get_parent()->get_node_id(node);

            if (node_id > 0 && node_id <= g->get_node_count())
            {
                auto node = g->get_node(node_id - 1);
                node->process(input_name, message);
            }
        }

    private:
        void topological_sort(std::vector<graph_node *> &nodes)
        {
            std::unordered_set<graph_node *> visited;
            std::vector<graph_node *> result;
            for (auto node : nodes)
            {
                dfs_postorder(node, visited, [&result](graph_node *node)
                              { result.push_back(node); });
            }
            std::reverse(result.begin(), result.end());
            nodes = result;
        }

        void initialize()
        {
            std::vector<graph_node *> nodes;
            for (uint32_t i = 0; i < g->get_node_count(); i++)
            {
                auto node = g->get_node(i);
                nodes.push_back(node.get());
            }

            topological_sort(nodes);

            for (auto node : nodes)
            {
                node->set_resources(resources);
                node->initialize();
            }
        }

        void finalize()
        {
            for (uint32_t i = 0; i < g->get_node_count(); i++)
            {
                auto node = g->get_node(i);
                node->finalize();
            }
        }

        void run(subgraph *g)
        {
            for (uint32_t i = 0; i < g->get_node_count(); i++)
            {
                auto node = g->get_node(i);
                node->run();
            }
        }

        void stop(subgraph *g)
        {
            for (uint32_t i = 0; i < g->get_node_count(); i++)
            {
                auto node = g->get_node(i);
                node->stop();
            }
        }
    };

    class graph_proc_server
    {
        rpc_server rpc_server_;
        std::shared_ptr<resource_list> resources_;
        std::map<uint32_t, std::shared_ptr<subgraph>> graphs_;
        std::mutex mtx;

    public:
        graph_proc_server(boost::asio::io_service &io_service, std::string address, uint16_t port, const std::shared_ptr<resource_list> &resources = std::make_shared<resource_list>())
            : rpc_server_(io_service, address, port), resources_(resources)
        {
            rpc_server_.register_handler((uint32_t)GRAPH_PROC_RPC_FUNC::DEPLOY, [this](uint32_t session, const std::vector<uint8_t> &arg, std::vector<uint8_t> &res) -> uint32_t
                                         {
            spdlog::debug("Deploy graph (session = {0})", session);

            std::stringstream ss(std::string(arg.begin(), arg.end()));
            std::shared_ptr<subgraph> g_(new subgraph());
            g_->load_from(ss);

            {
                std::lock_guard<std::mutex> lock(mtx);
                graphs_.insert(std::make_pair(session, g_));
            }

            return 0; });

            rpc_server_.register_handler((uint32_t)GRAPH_PROC_RPC_FUNC::INITIALIZE, [this](uint32_t session, const std::vector<uint8_t> &arg, std::vector<uint8_t> &res) -> uint32_t
                                         {
            spdlog::debug("Initialize graph (session = {0})", session);
            
            std::stringstream arg_ss(std::string((const char *)arg.data(), arg.size()));
            auto node_id = read_uint32(arg_ss);

            auto g_ = graphs_.at(session);

            assert(node_id > 0);
            auto node = g_->get_node(node_id - 1);

            std::unordered_map<std::string, subscribe_request> output_reqs;
            {
                cereal::BinaryInputArchive iarchive(arg_ss);
                iarchive(output_reqs);
            }
            
            std::string output_name;
            subscribe_request req;
            for (auto output_req: output_reqs)
            {
                std::tie(output_name, req) = output_req;
                auto output = node->get_output(output_name);
                output->request = req;
            }

            node->set_resources(resources_);
            node->initialize();

            std::unordered_map<std::string, subscribe_request> input_reqs;
            std::string input_name;
            graph_edge_ptr input_edge;
            for (auto input: node->get_inputs())
            {
                std::tie(input_name, input_edge) = input;
                auto req = input_edge->request;
                input_reqs.insert(std::make_pair(input_name, req));
            }
            
            std::stringstream res_ss;
            {
                cereal::BinaryOutputArchive oarchive(res_ss);
                oarchive(input_reqs);
            }

            std::string res_s = res_ss.str();
            std::copy(res_s.begin(), res_s.end(), std::back_inserter(res));

            return 0; });

            rpc_server_.register_handler((uint32_t)GRAPH_PROC_RPC_FUNC::RUN, [this](uint32_t session, const std::vector<uint8_t> &arg, std::vector<uint8_t> &res) -> uint32_t
                                         {
            spdlog::debug("Run graph (session = {0})", session);

            auto g_ = graphs_.at(session);
            g_->run();

            return 0; });

            rpc_server_.register_handler((uint32_t)GRAPH_PROC_RPC_FUNC::STOP, [this](uint32_t session, const std::vector<uint8_t> &arg, std::vector<uint8_t> &res) -> uint32_t
                                         {
            spdlog::debug("Stop graph (session = {0})", session);

            auto g_ = graphs_.at(session);
            g_->stop();

            return 0; });

            rpc_server_.register_handler((uint32_t)GRAPH_PROC_RPC_FUNC::FINALIZE, [this](uint32_t session, const std::vector<uint8_t> &arg, std::vector<uint8_t> &res) -> uint32_t
                                         {
            spdlog::debug("Finalize graph (session = {0})", session);
            
            std::stringstream arg_ss(std::string((const char *)arg.data(), arg.size()));
            auto node_id = read_uint32(arg_ss);

            auto g_ = graphs_.at(session);

            assert(node_id > 0);
            auto node = g_->get_node(node_id - 1);

            std::unordered_map<std::string, subscribe_request> output_reqs;
            {
                cereal::BinaryInputArchive iarchive(arg_ss);
                iarchive(output_reqs);
            }
            
            std::string output_name;
            subscribe_request req;
            for (auto output_req: output_reqs)
            {
                std::tie(output_name, req) = output_req;
                auto output = node->get_output(output_name);
                output->request = req;
            }

            node->finalize();

            std::unordered_map<std::string, subscribe_request> input_reqs;
            std::string input_name;
            graph_edge_ptr input_edge;
            for (auto input: node->get_inputs())
            {
                std::tie(input_name, input_edge) = input;
                auto req = input_edge->request;
                input_reqs.insert(std::make_pair(input_name, req));
            }
            
            std::stringstream res_ss;
            {
                cereal::BinaryOutputArchive oarchive(res_ss);
                oarchive(input_reqs);
            }

            std::string res_s = res_ss.str();
            std::copy(res_s.begin(), res_s.end(), std::back_inserter(res));

            return 0; });

            rpc_server_.register_handler((uint32_t)GRAPH_PROC_RPC_FUNC::PROCESS, [this](uint32_t session, const std::vector<uint8_t> &arg, std::vector<uint8_t> &res) -> uint32_t
                                         {
            
            std::stringstream arg_ss(std::string((const char *)arg.data(), arg.size()));
            const auto node_id = read_uint32(arg_ss);
            const auto input_name = read_string(arg_ss);
            spdlog::debug("Process node (session = {0}, node = {1}, input_name = {2})", session, node_id, input_name);

            std::shared_ptr<graph_message> msg;
            {
                cereal::BinaryInputArchive iarchive(arg_ss);
                iarchive(msg);
            }

            auto g_ = graphs_.at(session);

            assert(node_id > 0);
            auto node = g_->get_node(node_id - 1);

            node->process(input_name, msg);

            return 0; });

            rpc_server_.on_discconect([this](uint32_t session)
                                      {
            spdlog::debug("Delete graph (session = {0})", session);

            {
                std::lock_guard<std::mutex> lock(mtx);
                auto it = graphs_.find(session);
                if (it != graphs_.end())
                {
                    it->second->stop();
                    graphs_.erase(it);
                }
            } });
        }

        uint16_t get_port() const
        {
            return rpc_server_.local_endpoint().port();
        }
    };

    class graph_proc_client
    {
        std::vector<std::shared_ptr<subgraph>> graphs_;
        std::unordered_map<subgraph *, std::shared_ptr<rpc_client>> rpcs_;

    public:
        void deploy(asio::io_service &io_service, std::string ipaddress, uint16_t port, std::shared_ptr<subgraph> g)
        {
            g->validate();

            std::shared_ptr<rpc_client> rpc(new rpc_client(io_service));
            graphs_.push_back(g);
            rpcs_.insert(std::make_pair(g.get(), rpc));

            rpc->connect(ipaddress, port);
            invoke_deploy(*rpc, *g);
        }

        void run()
        {
            initialize();

            for (auto g : graphs_)
            {
                invoke_run_graph(g.get());
            }
        }

        void stop()
        {
            for (auto g : graphs_)
            {
                invoke_stop_graph(g.get());
            }

            finalize();
        }

        void process(const graph_node *node, const graph_message_ptr &message)
        {
            process(node, "default", message);
        }

        void process(const graph_node *node, const std::string &input_name, const graph_message_ptr &message)
        {
            auto g = node->get_parent();
            auto node_idx = node->get_parent()->get_node_id(node);

            invoke_process(g, node_idx, input_name, message);
        }

    private:
        void topological_sort(std::vector<graph_node *> &nodes)
        {
            std::unordered_set<graph_node *> visited;
            std::vector<graph_node *> result;
            for (auto node : nodes)
            {
                dfs_postorder(node, visited, [&result](graph_node *node)
                              { result.push_back(node); });
            }
            std::reverse(result.begin(), result.end());
            nodes = result;
        }

        void initialize()
        {
            std::vector<graph_node *> nodes;
            for (auto g : graphs_)
            {
                for (uint32_t i = 0; i < g->get_node_count(); i++)
                {
                    auto node = g->get_node(i);
                    nodes.push_back(node.get());
                }
            }

            topological_sort(nodes);

            for (auto node : nodes)
            {
                auto g = node->get_parent();
                auto node_idx = node->get_parent()->get_node_id(node);
                invoke_initialize_node(g, node_idx);
            }
        }

        void finalize()
        {
            std::vector<graph_node *> nodes;
            for (auto g : graphs_)
            {
                for (uint32_t i = 0; i < g->get_node_count(); i++)
                {
                    auto node = g->get_node(i);
                    nodes.push_back(node.get());
                }
            }

            topological_sort(nodes);

            for (auto node : nodes)
            {
                auto g = node->get_parent();
                auto node_idx = node->get_parent()->get_node_id(node);
                invoke_finalize_node(g, node_idx);
            }
        }

        void invoke_deploy(rpc_client &rpc, subgraph &g)
        {
            std::vector<uint8_t> arg, res;

            {
                std::stringstream output;
                g.save_to(output);

                std::string str = output.str();
                std::copy(str.begin(), str.end(), std::back_inserter(arg));
            }

            rpc.invoke((uint32_t)GRAPH_PROC_RPC_FUNC::DEPLOY, arg, res);
        }

        void invoke_initialize_node(subgraph *g, uint32_t node_id)
        {
            std::vector<uint8_t> arg, res;

            {
                std::stringstream output;
                write_uint32(output, node_id);

                std::unordered_map<std::string, subscribe_request> output_req;
                assert(node_id > 0);
                auto node = g->get_node(node_id - 1);
                std::string output_name;
                graph_edge_ptr output_edge;
                for (auto output : node->get_outputs())
                {
                    std::tie(output_name, output_edge) = output;
                    auto req = output_edge->request;
                    output_req.insert(std::make_pair(output_name, req));
                }
                {
                    cereal::BinaryOutputArchive oarchive(output);
                    oarchive(output_req);
                }

                std::string str = output.str();
                std::copy(str.begin(), str.end(), std::back_inserter(arg));
            }

            auto rpc = rpcs_.at(g);
            rpc->invoke((uint32_t)GRAPH_PROC_RPC_FUNC::INITIALIZE, arg, res);
            {
                std::stringstream input(std::string((const char *)res.data(), res.size()));

                std::unordered_map<std::string, subscribe_request> input_reqs;
                {
                    cereal::BinaryInputArchive iarchive(input);
                    iarchive(input_reqs);
                }

                assert(node_id > 0);
                auto node = g->get_node(node_id - 1);
                std::string input_name;
                subscribe_request req;
                for (auto input_req : input_reqs)
                {
                    std::tie(input_name, req) = input_req;
                    auto input_edge = node->get_input(input_name);
                    input_edge->request = req;
                }
            }
        }

        void invoke_finalize_node(subgraph *g, uint32_t node_id)
        {
            std::vector<uint8_t> arg, res;

            {
                std::stringstream output;
                write_uint32(output, node_id);

                std::unordered_map<std::string, subscribe_request> output_req;
                assert(node_id > 0);
                auto node = g->get_node(node_id - 1);
                std::string output_name;
                graph_edge_ptr output_edge;
                for (auto output : node->get_outputs())
                {
                    std::tie(output_name, output_edge) = output;
                    auto req = output_edge->request;
                    output_req.insert(std::make_pair(output_name, req));
                }
                {
                    cereal::BinaryOutputArchive oarchive(output);
                    oarchive(output_req);
                }

                std::string str = output.str();
                std::copy(str.begin(), str.end(), std::back_inserter(arg));
            }

            auto rpc = rpcs_.at(g);
            rpc->invoke((uint32_t)GRAPH_PROC_RPC_FUNC::FINALIZE, arg, res);
            {
                std::stringstream input(std::string((const char *)res.data(), res.size()));

                std::unordered_map<std::string, subscribe_request> input_reqs;
                {
                    cereal::BinaryInputArchive iarchive(input);
                    iarchive(input_reqs);
                }

                assert(node_id > 0);
                auto node = g->get_node(node_id - 1);
                std::string input_name;
                subscribe_request req;
                for (auto input_req : input_reqs)
                {
                    std::tie(input_name, req) = input_req;
                    auto input_edge = node->get_input(input_name);
                    input_edge->request = req;
                }
            }
        }

        void invoke_run_graph(subgraph *g)
        {
            std::vector<uint8_t> arg, res;

            auto rpc = rpcs_.at(g);
            rpc->invoke((uint32_t)GRAPH_PROC_RPC_FUNC::RUN, arg, res);
        }

        void invoke_stop_graph(subgraph *g)
        {
            std::vector<uint8_t> arg, res;

            auto rpc = rpcs_.at(g);
            rpc->invoke((uint32_t)GRAPH_PROC_RPC_FUNC::STOP, arg, res);
        }

        void invoke_process(subgraph *g, uint32_t node_id, const std::string &input_name, const graph_message_ptr &message)
        {
            std::vector<uint8_t> arg, res;

            {
                std::stringstream output;
                write_uint32(output, node_id);
                write_string(output, input_name);

                {
                    cereal::BinaryOutputArchive oarchive(output);
                    oarchive(message);
                }

                std::string str = output.str();
                std::copy(str.begin(), str.end(), std::back_inserter(arg));
            }

            auto rpc = rpcs_.at(g);
            rpc->invoke((uint32_t)GRAPH_PROC_RPC_FUNC::PROCESS, arg, res);
        }
    };

    // message implementations

    class text_message : public graph_message
    {
        std::string text;

    public:
        text_message()
            : text()
        {
        }

        void set_text(std::string text)
        {
            this->text = text;
        }
        std::string get_text() const
        {
            return text;
        }
        static std::string get_type()
        {
            return "text";
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(text);
        }
    };

    class number_message : public graph_message
    {
        double value;

    public:
        number_message()
            : value(0.0)
        {
        }

        void set_value(double value)
        {
            this->value = value;
        }
        double get_value() const
        {
            return value;
        }
        static std::string get_type()
        {
            return "number";
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(value);
        }
    };

    class list_message : public graph_message
    {
        std::vector<graph_message_ptr> list;

    public:
        list_message()
            : list()
        {
        }

        void add(graph_message_ptr value)
        {
            list.push_back(value);
        }
        graph_message_ptr get(size_t idx) const
        {
            return list[idx];
        }
        void set(size_t idx, graph_message_ptr value)
        {
            list[idx] = value;
        }
        std::size_t length() const
        {
            return list.size();
        }
        static std::string get_type()
        {
            return "list";
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(list);
        }
    };

    class object_message : public graph_message
    {
        std::unordered_map<std::string, graph_message_ptr> fields;

    public:
        object_message()
            : fields()
        {
        }

        void add_field(std::string name, graph_message_ptr value)
        {
            fields.insert(std::make_pair(name, value));
        }
        graph_message_ptr get_field(std::string name) const
        {
            return fields.at(name);
        }
        void set_field(std::string name, graph_message_ptr value)
        {
            fields[name] = value;
        }
        const std::unordered_map<std::string, graph_message_ptr> &get_fields() const
        {
            return fields;
        }

        static std::string get_type()
        {
            return "object";
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(fields);
        }
    };

    class blob_message : public graph_message
    {
        std::vector<uint8_t> data;

    public:
        blob_message()
            : data()
        {
        }

        void set_data(const std::vector<uint8_t> &data)
        {
            this->data = data;
        }
        void set_data(std::vector<uint8_t> &&data)
        {
            this->data = std::move(data);
        }
        const std::vector<uint8_t> &get_data() const
        {
            return data;
        }
        static std::string get_type()
        {
            return "blob";
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(data);
        }
    };

    // node implementations

    class passthrough_node : public graph_node
    {
        graph_edge_ptr output;

    public:
        passthrough_node()
            : graph_node(), output(std::make_shared<graph_edge>(this))
        {
            set_output(output);
        }

        virtual std::string get_proc_name() const override
        {
            return "passthrough";
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
        }

        virtual void process(std::string input_name, graph_message_ptr message) override
        {
            output->send(message);
        }
    };

    class p2p_talker_node : public graph_node
    {
        graph_edge_ptr output;
        std::shared_ptr<data_stream_transmitter> transmitter;
        boost::asio::io_service io_service;
        std::shared_ptr<std::thread> th;
        std::atomic_bool running;

    public:
        p2p_talker_node()
            : graph_node(), output(std::make_shared<graph_edge>(this, EDGE_TYPE::CHAIN)), transmitter(), io_service(), th(), running(false)
        {
            set_output(output);
        }

        virtual ~p2p_talker_node()
        {
            finalize();
        }

        virtual void finalize() override
        {
            if (transmitter)
            {
                transmitter->close();
                transmitter.reset();
            }
        }

        virtual void initialize() override
        {
            auto output_req = get_output()->request;
            auto data = output_req.get_data();
            std::stringstream ss(std::string(data.begin(), data.end()));

            auto remote_address = read_string(ss);
            auto remote_port = read_uint16(ss);

            transmitter.reset(new data_stream_transmitter(io_service));
            transmitter->open(remote_address, remote_port);
        }

        virtual void process(std::string input_name, graph_message_ptr message) override
        {
            source_identifier id{0, 0};
            std::stringstream ss;
            {
                cereal::BinaryOutputArchive oarchive(ss);
                oarchive(message);
            }
            std::string str = ss.str();

            auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();

            transmitter->send(id, (double)(ns / 100), (uint8_t *)str.data(), str.size());
        }

        virtual std::string get_proc_name() const override
        {
            return "data_talker";
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
        }

        virtual void run() override
        {
            running = true;
            th.reset(new std::thread([&]()
                                     { io_service.run(); }));
        }

        virtual void stop() override
        {
            if (running.load())
            {
                running.store(false);
                io_service.stop();
                if (th && th->joinable())
                {
                    th->join();
                }
            }
        }
    };

    class p2p_tcp_talker_node : public graph_node
    {
        graph_edge_ptr output;
        std::shared_ptr<data_stream_tcp_transmitter> transmitter;
        boost::asio::io_service io_service;
        std::shared_ptr<std::thread> th;
        std::atomic_bool running;

    public:

        p2p_tcp_talker_node()
            : graph_node(), output(std::make_shared<graph_edge>(this, EDGE_TYPE::CHAIN)), transmitter()
        {
            set_output(output);
        }

        virtual ~p2p_tcp_talker_node()
        {
            finalize();
        }

        virtual void finalize() override
        {
            if (transmitter)
            {
                transmitter->close();
                transmitter.reset();
            }
        }

        virtual void initialize() override
        {
            const auto output_req = get_output()->request;
            const auto data = output_req.get_data();
            std::stringstream ss(std::string(data.begin(), data.end()));

            const auto address = read_string(ss);
            const auto port = read_uint16(ss);

            transmitter.reset(new data_stream_tcp_transmitter(io_service));
            transmitter->open(address, port);
        }

        virtual void process(std::string input_name, graph_message_ptr message) override
        {
            source_identifier id{0, 0};
            std::stringstream ss;
            {
                cereal::BinaryOutputArchive oarchive(ss);
                oarchive(message);
            }
            std::string str = ss.str();

            auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();

            transmitter->send(id, (double)(ns / 100), (uint8_t *)str.data(), str.size());
        }

        virtual std::string get_proc_name() const override
        {
            return "data_tcp_talker";
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
        }

        virtual void run() override
        {
            running = true;
            th.reset(new std::thread([&]()
                                     { io_service.run(); }));
        }

        virtual void stop() override
        {
            if (running.load())
            {
                running.store(false);
                io_service.stop();
                if (th && th->joinable())
                {
                    th->join();
                }
            }
        }
    };

    class p2p_listener_node : public graph_node
    {
        graph_edge_ptr output;
        std::shared_ptr<data_stream_receiver> receiver;
        std::string address;
        uint16_t port;

    public:
        p2p_listener_node()
            : graph_node(), output(std::make_shared<graph_edge>(this))
        {
            set_output(output);
        }

        virtual std::string get_proc_name() const override
        {
            return "data_listener";
        }

        void set_endpoint(std::string address, uint16_t port)
        {
            this->address = address;
            this->port = port;
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(address);
            archive(port);
        }

        virtual void initialize() override
        {
            receiver.reset(new data_stream_receiver(udp::endpoint(udp::v4(), port)));

            source_identifier id{0, 0};
            receiver->add_session(id);

            auto bind_port = receiver->local_endpoint().port();
            auto bind_address = receiver->local_endpoint().address().to_string();

            std::stringstream ss;
            write_string(ss, address);
            write_uint16(ss, bind_port);

            std::string str = ss.str();
            std::vector<uint8_t> data(str.begin(), str.end());

            subscribe_request req;
            auto input = get_input();
            req.set_proc_name(get_proc_name());
            req.set_msg_type(text_message::get_type());
            req.set_data(data);
            input->request = req;
        }

        virtual void run() override
        {
            receiver->start([this](double timestamp, source_identifier id, asio::streambuf &stream)
                            { this->on_receive_handler(timestamp, id, stream); });
        }

        virtual void stop() override
        {
            receiver->stop();
        }

        void on_receive_handler(double timestamp, source_identifier id, asio::streambuf &stream)
        {
            if (id.data_id == 0)
            {
                on_receive_data_handler(timestamp, id, stream);
            }
            else
            {
                spdlog::error("Received unknown data");
            }
        }

        void on_receive_data_handler(double timestamp, source_identifier id, asio::streambuf &stream)
        {
            if (stream.size() < sizeof(int))
            {
                return;
            }

            std::string str(boost::asio::buffers_begin(stream.data()), boost::asio::buffers_end(stream.data()));
            std::stringstream ss(str);

            std::shared_ptr<graph_message> msg;
            {
                cereal::BinaryInputArchive iarchive(ss);
                iarchive(msg);
            }

            output->send(msg);
        }
    };

    class p2p_tcp_listener_node : public graph_node
    {
        graph_edge_ptr output;
        std::shared_ptr<data_stream_tcp_receiver> receiver;
        std::string address;
        uint16_t port;

    public:
        p2p_tcp_listener_node()
            : graph_node(), output(std::make_shared<graph_edge>(this))
        {
            set_output(output);
        }

        virtual std::string get_proc_name() const override
        {
            return "data_tcp_listener";
        }

        void set_endpoint(std::string address, uint16_t port)
        {
            this->address = address;
            this->port = port;
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(address);
            archive(port);
        }

        virtual void initialize() override
        {
            receiver.reset(new data_stream_tcp_receiver(tcp::endpoint(asio::ip::address_v4::from_string(address), port)));

            const source_identifier id{0, 0};
            receiver->add_session(id);

            const auto bind_port = receiver->local_endpoint().port();
            const auto bind_address = receiver->local_endpoint().address().to_string();

            std::stringstream ss;
            write_string(ss, address);
            write_uint16(ss, bind_port);

            const auto str = ss.str();
            const std::vector<uint8_t> data(str.begin(), str.end());

            subscribe_request req;
            const auto input = get_input();
            req.set_proc_name(get_proc_name());
            req.set_msg_type(text_message::get_type());
            req.set_data(data);
            input->request = req;
        }

        virtual void run() override
        {
            receiver->start([this](double timestamp, source_identifier id, asio::streambuf &stream)
                            { this->on_receive_handler(timestamp, id, stream); });
        }

        virtual void stop() override
        {
            receiver->stop();
        }

        void on_receive_handler(double timestamp, source_identifier id, asio::streambuf &stream)
        {
            if (id.data_id == 0)
            {
                on_receive_data_handler(timestamp, id, stream);
            }
            else
            {
                spdlog::error("Received unknown data");
            }
        }

        void on_receive_data_handler(double timestamp, source_identifier id, asio::streambuf &stream)
        {
            if (stream.size() < sizeof(int))
            {
                return;
            }

            std::string str(boost::asio::buffers_begin(stream.data()), boost::asio::buffers_end(stream.data()));
            std::stringstream ss(str);

            std::shared_ptr<graph_message> msg;
            {
                cereal::BinaryInputArchive iarchive(ss);
                iarchive(msg);
            }

            output->send(msg);
        }
    };

    class broadcast_talker_node : public graph_node
    {
        graph_edge_ptr output;
        std::shared_ptr<data_stream_transmitter> transmitter;
        boost::asio::io_service io_service;
        std::shared_ptr<std::thread> th;
        std::atomic_bool running;
        std::string address;
        std::uint16_t port;

    public:
        broadcast_talker_node()
            : graph_node(), output(std::make_shared<graph_edge>(this, EDGE_TYPE::CHAIN)), transmitter(), io_service(), th(), running(false)
        {
            set_output(output);
        }

        void set_endpoint(std::string address, uint16_t port)
        {
            this->address = address;
            this->port = port;
        }

        virtual ~broadcast_talker_node()
        {
            finalize();
        }

        virtual void finalize() override
        {
            if (transmitter)
            {
                transmitter->close();
                transmitter.reset();
            }
        }

        virtual void initialize() override
        {
            transmitter.reset(new data_stream_transmitter(io_service));

            if (asio::ip::address_v4::from_string(address).is_multicast())
            {
                transmitter->open(address, port);
            }
            else
            {
                transmitter->open_broadcast(port);
            }
        }

        virtual void process(std::string input_name, graph_message_ptr message) override
        {
            source_identifier id{0, 0};
            std::stringstream ss;
            {
                cereal::BinaryOutputArchive oarchive(ss);
                oarchive(message);
            }
            std::string str = ss.str();

            auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();

            transmitter->send(id, (double)(ns / 100), (uint8_t *)str.data(), str.size());
        }

        virtual std::string get_proc_name() const override
        {
            return "broadcast_talker";
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(address);
            archive(port);
        }

        virtual void run() override
        {
            running = true;
            th.reset(new std::thread([&]()
                                     { io_service.run(); }));
        }

        virtual void stop() override
        {
            if (running.load())
            {
                running.store(false);
                io_service.stop();
                if (th && th->joinable())
                {
                    th->join();
                }
            }
        }
    };

    class broadcast_listener_node : public graph_node
    {
        graph_edge_ptr output;
        std::shared_ptr<data_stream_receiver> receiver;
        std::string address;
        uint16_t port;
        std::string multicast_address;

    public:
        broadcast_listener_node()
            : graph_node(), output(std::make_shared<graph_edge>(this))
        {
            set_output(output);
        }

        virtual std::string get_proc_name() const override
        {
            return "broadcast_listener";
        }

        void set_endpoint(std::string address, uint16_t port, std::string multicast_address = std::string())
        {
            this->address = address;
            this->port = port;
            this->multicast_address = multicast_address;
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(address);
            archive(port);
            archive(multicast_address);
        }

        virtual void initialize() override
        {
            if (!multicast_address.empty())
            {
                receiver.reset(new data_stream_receiver(udp::endpoint(udp::v4(), port), multicast_address));
            }
            else
            {
                receiver.reset(new data_stream_receiver(udp::endpoint(udp::v4(), port), true));
            }

            source_identifier id{0, 0};
            receiver->add_session(id);
        }

        virtual void run() override
        {
            receiver->start([this](double timestamp, source_identifier id, asio::streambuf &stream)
                            { this->on_receive_handler(timestamp, id, stream); });
        }

        virtual void stop() override
        {
            receiver->stop();
        }

        void on_receive_handler(double timestamp, source_identifier id, asio::streambuf &stream)
        {
            if (id.data_id == 0)
            {
                on_receive_data_handler(timestamp, id, stream);
            }
            else
            {
                spdlog::error("Received unknown data");
            }
        }

        void on_receive_data_handler(double timestamp, source_identifier id, asio::streambuf &stream)
        {
            if (stream.size() < sizeof(int))
            {
                return;
            }

            std::string str(boost::asio::buffers_begin(stream.data()), boost::asio::buffers_end(stream.data()));
            std::stringstream ss(str);

            std::shared_ptr<graph_message> msg;
            {
                cereal::BinaryInputArchive iarchive(ss);
                iarchive(msg);
            }

            output->send(msg);
        }
    };

    class heartbeat_node : public graph_node
    {
        uint32_t interval;
        std::shared_ptr<std::thread> th;
        std::atomic_bool running;

    public:
        heartbeat_node()
            : graph_node(), interval(1000), running(false)
        {
        }

        virtual std::string get_proc_name() const override
        {
            return "heartbeat";
        }

        void set_interval(uint32_t interval)
        {
            this->interval = interval;
        }

        uint32_t get_interval() const
        {
            return interval;
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(interval);
        }

        virtual void run() override
        {
            running = true;
            th.reset(new std::thread([this]()
                                     {
                while (running.load())
                {
                    tick();
                    std::this_thread::sleep_for(std::chrono::milliseconds(interval));
                } }));
        }

        virtual void stop() override
        {
            if (running.load())
            {
                running.store(false);
                if (th && th->joinable())
                {
                    th->join();
                }
            }
        }

        virtual void tick()
        {
        }
    };

    class text_heartbeat_node : public heartbeat_node
    {
        std::string message;
        graph_edge_ptr output;

    public:
        text_heartbeat_node()
            : heartbeat_node(), message(), output(std::make_shared<graph_edge>(this))
        {
            set_output(output);
        }

        void set_message(std::string message)
        {
            this->message = message;
        }

        std::string get_message() const
        {
            return message;
        }

        virtual std::string get_proc_name() const override
        {
            return "text_heartbeat";
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(cereal::base_class<heartbeat_node>(this));
            archive(message);
        }

        virtual void tick() override
        {
            auto msg = std::make_shared<text_message>();
            msg->set_text(message);
            output->send(msg);
        }
    };

    class console_node : public graph_node
    {
        std::ostream *output;

    public:
        console_node()
            : graph_node(), output(&std::cout)
        {
        }

        virtual std::string get_proc_name() const override
        {
            return "console";
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
        }

        virtual void run() override
        {
        }

        virtual void stop() override
        {
        }

        virtual void process(std::string input_name, graph_message_ptr message) override
        {
            if (auto text = std::dynamic_pointer_cast<text_message>(message))
            {
                (*output) << text->get_text();
            }
        }
    };

    class buffer_node : public heartbeat_node
    {
        graph_edge_ptr output;
        graph_message_ptr message;
        std::mutex mtx;

    public:
        buffer_node()
            : heartbeat_node(), output(std::make_shared<graph_edge>(this))
        {
            set_output(output);
        }

        virtual std::string get_proc_name() const override
        {
            return "buffer";
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(cereal::base_class<heartbeat_node>(this));
        }

        virtual void process(std::string input_name, graph_message_ptr message) override
        {
            std::lock_guard<std::mutex> lock(mtx);

            if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
            {
                if (!this->message)
                {
                    this->message = message;
                }
                else
                {
                    auto current_obj_msg = std::dynamic_pointer_cast<object_message>(this->message);
                    for (auto &[name, field] : obj_msg->get_fields())
                    {
                        current_obj_msg->set_field(name, field);
                    }
                }
            }
            else if (auto list_msg = std::dynamic_pointer_cast<list_message>(message))
            {
                if (!this->message)
                {
                    this->message = message;
                }
                else
                {
                    auto current_list_msg = std::dynamic_pointer_cast<list_message>(this->message);

                    const auto copy_size = std::min(current_list_msg->length(), list_msg->length());
                    std::size_t i = 0;
                    for (; i < copy_size; i++)
                    {
                        current_list_msg->set(i, list_msg->get(i));
                    }
                    for (; i < list_msg->length(); i++)
                    {
                        current_list_msg->add(list_msg->get(i));
                    }
                }
            }
            else
            {
                this->message = message;
            }
        }

        virtual void tick() override
        {
            std::lock_guard<std::mutex> lock(mtx);
            output->send(message);
        }
    };

    class clock_buffer_node : public graph_node
    {
        graph_edge_ptr output;
        graph_message_ptr message;
        std::mutex mtx;

    protected:
        graph_edge_ptr get_clock() const
        {
            return get_input("clock");
        }

    public:
        clock_buffer_node()
            : graph_node(), output(std::make_shared<graph_edge>(this))
        {
            set_output(output);
        }

        void set_clock(graph_edge_ptr clock)
        {
            set_input(clock, "clock");
        }

        virtual std::string get_proc_name() const override
        {
            return "clock_buffer";
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
        }

        virtual void process(std::string input_name, graph_message_ptr message) override
        {
            std::lock_guard<std::mutex> lock(mtx);

            if (input_name == "default")
            {
                this->message = message;
            }
            else if (input_name == "clock")
            {
                output->send(this->message);
            }
        }
    };

    class mux_node : public graph_node
    {
        graph_edge_ptr output;

    public:
        mux_node()
            : graph_node(), output(std::make_shared<graph_edge>(this))
        {
            set_output(output);
        }

        virtual std::string get_proc_name() const override
        {
            return "mux_node";
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
        }

        virtual void process(std::string input_name, graph_message_ptr message) override
        {
            auto msg = std::make_shared<object_message>();
            msg->add_field(input_name, message);
            output->send(msg);
        }
    };

    class demux_node : public graph_node
    {
    public:
        demux_node()
            : graph_node()
        {
        }

        graph_edge_ptr add_output(std::string name)
        {
            auto outputs = get_outputs();
            auto it = outputs.find(name);
            if (it == outputs.end())
            {
                auto output = std::make_shared<graph_edge>(this);
                set_output(output, name);
                return output;
            }
            return it->second;
        }

        virtual std::string get_proc_name() const override
        {
            return "demux_node";
        }

        template <typename Archive>
        void save(Archive &archive) const
        {
            std::vector<std::string> output_names;
            auto outputs = get_outputs();
            for (auto output : outputs)
            {
                output_names.push_back(output.first);
            }
            archive(output_names);
        }

        template <typename Archive>
        void load(Archive &archive)
        {
            std::vector<std::string> output_names;
            archive(output_names);
            for (auto output_name : output_names)
            {
                set_output(std::make_shared<graph_edge>(this), output_name);
            }
        }

        virtual void process(std::string input_name, graph_message_ptr message) override
        {
            if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
            {
                for (auto field : obj_msg->get_fields())
                {
                    try
                    {
                        get_output(field.first)->send(field.second);
                    }
                    catch (const std::invalid_argument &e)
                    {
                        spdlog::warn(e.what());
                    }
                }
            }
        }
    };

    class fifo_node : public graph_node
    {
        graph_edge_ptr output;
        std::mutex mtx;
        std::deque<graph_message_ptr> messages;
        std::shared_ptr<std::thread> th;
        std::atomic_bool running;
        std::condition_variable cv;
        std::uint32_t max_size;

    public:
        fifo_node()
            : graph_node(), output(std::make_shared<graph_edge>(this)), max_size(10)
        {
            set_output(output);
        }

        virtual std::string get_proc_name() const override
        {
            return "fifo_node";
        }

        void set_max_size(std::uint32_t value)
        {
            max_size = value;
        }
        std::uint32_t get_max_size() const
        {
            return max_size;
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(max_size);
        }

        virtual void process(std::string input_name, graph_message_ptr message) override
        {
            if (!running)
            {
                return;
            }

            if (input_name == "default")
            {
                std::lock_guard<std::mutex> lock(mtx);

                if (messages.size() >= max_size)
                {
                    std::cout << "Fifo overflow" << std::endl;
                    spdlog::error("Fifo overflow");
                }
                else
                {
                    messages.push_back(message);
                    cv.notify_one();
                }
            }
        }

        virtual void run() override
        {
            running = true;
            th.reset(new std::thread([this]()
                                     {
                while (running.load())
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    cv.wait(lock, [&] { return !messages.empty() || !running; });

                    if (!running)
                    {
                        break;
                    }
                    if (!messages.empty())
                    {
                        const auto message = messages.front();
                        messages.pop_front();
                        output->send(message);
                    }
                } }));
        }

        virtual void stop() override
        {
            if (running.load())
            {
                running.store(false);
                cv.notify_one();
                if (th && th->joinable())
                {
                    th->join();
                }
            }
        }
    };
}

CEREAL_REGISTER_TYPE(coalsack::text_message)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_message, coalsack::text_message)

CEREAL_REGISTER_TYPE(coalsack::number_message)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_message, coalsack::number_message)

CEREAL_REGISTER_TYPE(coalsack::list_message)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_message, coalsack::list_message)

CEREAL_REGISTER_TYPE(coalsack::object_message)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_message, coalsack::object_message)

CEREAL_REGISTER_TYPE(coalsack::blob_message)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_message, coalsack::blob_message)

CEREAL_REGISTER_TYPE(coalsack::passthrough_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::passthrough_node)

CEREAL_REGISTER_TYPE(coalsack::p2p_talker_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::p2p_talker_node)

CEREAL_REGISTER_TYPE(coalsack::p2p_tcp_talker_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::p2p_tcp_talker_node)

CEREAL_REGISTER_TYPE(coalsack::p2p_listener_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::p2p_listener_node)

CEREAL_REGISTER_TYPE(coalsack::p2p_tcp_listener_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::p2p_tcp_listener_node)

CEREAL_REGISTER_TYPE(coalsack::broadcast_talker_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::broadcast_talker_node)

CEREAL_REGISTER_TYPE(coalsack::broadcast_listener_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::broadcast_listener_node)

CEREAL_REGISTER_TYPE(coalsack::heartbeat_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::heartbeat_node)

CEREAL_REGISTER_TYPE(coalsack::text_heartbeat_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::heartbeat_node, coalsack::text_heartbeat_node)

CEREAL_REGISTER_TYPE(coalsack::console_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::console_node)

CEREAL_REGISTER_TYPE(coalsack::buffer_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::heartbeat_node, coalsack::buffer_node)

CEREAL_REGISTER_TYPE(coalsack::clock_buffer_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::clock_buffer_node)

CEREAL_REGISTER_TYPE(coalsack::mux_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::mux_node)

CEREAL_REGISTER_TYPE(coalsack::demux_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::demux_node)

CEREAL_REGISTER_TYPE(coalsack::fifo_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::fifo_node)
