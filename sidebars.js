/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

 module.exports = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  sidebar: [
    "index",
    {
      type: "category",
      label: "介绍 （ Introduction ）",
      collapsed: false,
      collapsible: false,
      items: [{ type: "autogenerated", dirName: "langchain-intro" }],
      link: {
        type: 'generated-index',
	      slug: "components",
      },
    },
    {
      type: "category",
      label: "提示工程 （ Prompt Engineering ）",
      collapsed: false,
      collapsible: false,
      items: [{ type: "autogenerated", dirName: "langchain-prompt-templates" }],
      link: {
        type: 'generated-index',
	      slug: "components",
      },
    },
    {
      type: "category",
      label: "会话记忆 （ Conversational Memory ）",
      collapsed: false,
      collapsible: false,
      items: [{ type: "autogenerated", dirName: "langchain-conversational-memory" }],
      link: {
        type: 'generated-index',
	      slug: "components",
      },
    },
    {
      type: "category",
      label: "基本常识 （ Knowledge Bases ）",
      collapsed: false,
      collapsible: false,
      items: [{ type: "autogenerated", dirName: "langchain-retrieval-augmentation" }],
      link: {
        type: 'generated-index',
	      slug: "components",
      },
    },
    {
      type: "category",
      label: "会话代理 (Agents)  （ Conversational Agents ）",
      collapsed: false,
      collapsible: false,
      items: [{ type: "autogenerated", dirName: "langchain-agents" }],
      link: {
        type: 'generated-index',
	      slug: "components",
      },
    },
    {
      type: "category",
      label: "自定义工具 （ Custom Tools ）",
      collapsed: false,
      collapsible: false,
      items: [{ type: "autogenerated", dirName: "langchain-tools" }],
      link: {
        type: 'generated-index',
	      slug: "components",
      },
    },
    {
      type: "html",
      value: '<img src="https://pic1.zhimg.com/80/v2-31131dcb1732cb5bca7c182c9e8da046_r.jpg" alt="扫我，入群" width="280" height="330"/>'
    }
  ],
};
